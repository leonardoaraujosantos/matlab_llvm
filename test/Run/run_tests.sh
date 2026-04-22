#!/usr/bin/env bash
# Build-and-run tests. For each .m in this directory, compiles with matlabc
# + clang + the matlab runtime, runs the executable, and compares stdout to
# the matching .stdout file. Failure if stdout differs or exit is non-zero.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
RUNTIME="$ROOT/runtime/matlab_runtime.c"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $m (no .stdout)"; continue; }

  tmpll="$(mktemp -t mlc.XXXXXX).ll"
  tmpbin="$(mktemp -t mlc.XXXXXX).out"

  if ! "$MATLABC" -emit-llvm "$m" > "$tmpll" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-llvm errored"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  fi
  if ! "$CLANG" -Wno-override-module "$tmpll" "$RUNTIME" -o "$tmpbin" 2>/dev/null; then
    echo "FAIL $base: clang link failed"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  fi
  got="$("$tmpbin")" || {
    echo "FAIL $base: non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  }
  # If a .sorted file exists alongside the .m, compare against the expected
  # output after sorting both sides (useful for parfor where iteration
  # order is nondeterministic).
  if [[ -e "${m%.m}.sorted" ]]; then
    if diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch (sorted)"
      diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) | sed 's/^/  /'
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
  rm -f "$tmpll" "$tmpbin"
done

echo "----"
echo "run passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
