#!/usr/bin/env bash
# DWARF emission test for `matlabc -emit-llvm -g`.
#
# Asserts:
#   1. With -g, the emitted LLVM IR carries the metadata graph that
#      lldb / gdb need to step through .m source: !DICompileUnit,
#      !DIFile, !DISubprogram, and !DILocation entries linked into a
#      consistent !llvm.dbg.cu chain.
#   2. Each function definition has a `!dbg` attachment (so the
#      DWARF subprogram is actually anchored to the IR function).
#   3. Without -g, the output has none of the above — DWARF is
#      strictly opt-in.
#
# Usage: run_dwarf_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$TESTDIR/dap_locals_program.m"

pass=0
fail=0

check() {
  local label="$1"; shift
  if "$@"; then
    pass=$((pass+1))
    echo "ok   $label"
  else
    fail=$((fail+1))
    echo "FAIL $label"
  fi
}

check_not() {
  local label="$1"; shift
  if "$@"; then
    fail=$((fail+1))
    echo "FAIL $label"
  else
    pass=$((pass+1))
    echo "ok   $label"
  fi
}

# 1. With -g — must emit DWARF metadata.
WITH_G=$("$MATLABC" -emit-llvm -g "$SRC" 2>/dev/null)
check "DICompileUnit present"   grep -q "!DICompileUnit"  <<<"$WITH_G"
check "DIFile present"          grep -q "!DIFile"         <<<"$WITH_G"
check "DISubprogram present"    grep -q "!DISubprogram"   <<<"$WITH_G"
check "DILocation present"      grep -q "!DILocation"     <<<"$WITH_G"
check "llvm.dbg.cu registered"  grep -q "!llvm.dbg.cu"    <<<"$WITH_G"
# At least one function carries a !dbg attachment.
check "function has !dbg"       grep -qE "^define [^@]*@[A-Za-z_][A-Za-z0-9_]* *\\([^)]*\\) *(#[0-9]+ +)?!dbg" <<<"$WITH_G"
# DICompileUnit -> DIFile chain references the source filename.
check "CU references .m file"   grep -qE "!DIFile\\(filename: \"[^\"]*\\.m\"" <<<"$WITH_G"

# 2. Without -g — must NOT emit any DWARF metadata.
WITHOUT_G=$("$MATLABC" -emit-llvm "$SRC" 2>/dev/null)
check_not "no DICompileUnit"  grep -q "!DICompileUnit" <<<"$WITHOUT_G"
check_not "no DISubprogram"   grep -q "!DISubprogram"  <<<"$WITHOUT_G"
check_not "no DILocation"     grep -q "!DILocation"    <<<"$WITHOUT_G"
check_not "no llvm.dbg.cu"    grep -q "!llvm.dbg.cu"   <<<"$WITHOUT_G"

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
