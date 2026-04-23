#!/usr/bin/env bash
# Fail-fast tests for the C emission path. For each .m in this directory,
# `matlabc -emit-c` is expected to EXIT NON-ZERO and emit a stderr line
# that contains the matching .stderr file's content verbatim.
#
# This locks in the robustness contract: unsupported MLIR ops produce a
# clear diagnostic rather than silently generating broken C.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stderr"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .stderr)"; continue; }

  tmpout="$(mktemp -t mlc.XXXXXX).out"
  tmperr="$(mktemp -t mlc.XXXXXX).err"
  "$MATLABC" -emit-c "$m" >"$tmpout" 2>"$tmperr"
  rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "FAIL $base: matlabc -emit-c exited 0 (expected non-zero)"
    fail=$((fail+1))
    rm -f "$tmpout" "$tmperr"; continue
  fi

  expected="$(cat "$exp")"
  if ! grep -qF -- "$expected" "$tmperr"; then
    echo "FAIL $base: expected stderr substring not found"
    echo "  expected: $expected"
    echo "  stderr:"
    sed 's/^/    /' "$tmperr"
    fail=$((fail+1))
    rm -f "$tmpout" "$tmperr"; continue
  fi

  pass=$((pass+1))
  rm -f "$tmpout" "$tmperr"
done

echo "----"
echo "emit-c-fail passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
