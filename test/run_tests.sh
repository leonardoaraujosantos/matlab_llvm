#!/usr/bin/env bash
# Golden-file test runner.
#
# For each *.m file under Lexer/ and Parser/, runs matlabc with -dump-tokens
# or -dump-ast and diffs against the matching *.expected file.
#
# Usage: run_tests.sh <path-to-matlabc>
#        UPDATE=1 run_tests.sh <path-to-matlabc>   # refresh expected files
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TESTDIR"

pass=0
fail=0
failed_names=()

run_suite() {
  local dir="$1"
  local flag="$2"
  for m in "$dir"/*.m; do
    [[ -e "$m" ]] || continue
    local exp="${m%.m}.expected"
    local got
    # shellcheck disable=SC2086
    got="$("$MATLABC" $flag "$m" 2>&1)" || true
    if [[ -n "${UPDATE:-}" || ! -e "$exp" ]]; then
      printf '%s\n' "$got" > "$exp"
      echo "UPDATED $m"
      continue
    fi
    if diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      failed_names+=("$m")
      echo "FAIL $m"
      diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
    fi
  done
}

run_suite Lexer    -dump-tokens
run_suite Parser   -dump-ast
run_suite Sema     -emit-sema
run_suite MIR      -emit-mir
run_suite MLIR     -emit-mlir
run_suite Opt      "-emit-mlir -opt"
run_suite Programs "-emit-mlir -opt"

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
