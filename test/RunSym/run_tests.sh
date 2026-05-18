#!/usr/bin/env bash
# Symbolic Math Toolbox end-to-end tests.
#
# For each .m in this directory, drive matlabc -emit-cpp through the same
# clang link line as test/Run/run_tests.sh, but additionally link
# runtime/runtime_sym.cpp and SymPP. Compare stdout to .stdout.
#
# Usage: run_tests.sh <path-to-matlabc> [<sympp-prefix>]
#
# When <sympp-prefix> is omitted, falls back to /tmp/sympp_install (the
# default the project's CMakeLists.txt looks for) and finally to
# /opt/homebrew. Tests are skipped (rc=77, the autoconf "skipped" code)
# when neither SymPP install is found — matches the cocotb-tests pattern
# already used by CTest in this repo.
set -u

MATLABC="${1:-}"
SYMPP_PREFIX="${2:-${SYMPP_PREFIX:-/tmp/sympp_install}}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc> [<sympp-prefix>]" >&2
  exit 2
fi

if [[ ! -e "$SYMPP_PREFIX/include/sympp/sympp.hpp" ]]; then
  echo "skip: SymPP not found at $SYMPP_PREFIX (set SYMPP_PREFIX or pass as 2nd arg)" >&2
  exit 77
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
CXX="${CXX:-${CLANG}++}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

# Runtime — shares matlab_runtime.cpp / runtime_debug.cpp / runtime_complex.cpp
# with test/Run plus runtime_sym.cpp for the sym entry points.
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
  "$ROOT/runtime/toolbox/sym/runtime_sym.cpp"
)

pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $m (no .stdout)"; continue; }

  tmpcpp="$(mktemp -t mlcsym.XXXXXX).cpp"
  tmpbin="$(mktemp -t mlcsym.XXXXXX).out"

  if ! "$MATLABC" -emit-cpp "$m" > "$tmpcpp" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-cpp errored"
    fail=$((fail+1))
    rm -f "$tmpcpp" "$tmpbin"; continue
  fi
  if ! "$CXX" -std=c++20 -DMATLAB_LLVM_WITH_SYM=1 \
        "$tmpcpp" "${RUNTIME_SRCS[@]}" \
        -I"$ROOT/runtime" -I"$SYMPP_PREFIX/include" -I/opt/homebrew/include \
        -L"$SYMPP_PREFIX/lib" -lsympp \
        -L/opt/homebrew/lib -lgmp -lmpfr -lgmpxx \
        -Wl,-rpath,"$SYMPP_PREFIX/lib" \
        -o "$tmpbin" 2>/tmp/mlcsym.err; then
    echo "FAIL $base: clang link failed"
    cat /tmp/mlcsym.err | sed 's/^/  /' | head -10
    fail=$((fail+1))
    rm -f "$tmpcpp" "$tmpbin"; continue
  fi
  got="$("$tmpbin")" || {
    echo "FAIL $base: non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpcpp" "$tmpbin"; continue
  }
  if diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
  rm -f "$tmpcpp" "$tmpbin"
done

echo "----"
echo "sym passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
