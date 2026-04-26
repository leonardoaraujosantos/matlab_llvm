#!/usr/bin/env bash
# Debug-hook test runner.
#
# For each *.m fixture in this directory:
#   1. Compile with `matlabc -emit-mlir -g` (the -g flag flips DebugMode
#      on in lowering, so every statement gets a matlab_dbg_hook call
#      injected just like the -dap path does).
#   2. Extract the line operand passed to each matlab_dbg_hook call
#      from the MLIR text dump.
#   3. Diff the extracted list against the matching *.expected file.
#   4. Run a property check confirming every hook line points at a
#      non-blank, non-comment-only source row.
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

EXTRACT="$TESTDIR/extract_hook_lines.py"
CHECK="$TESTDIR/check_executable_lines.py"

pass=0
fail=0
failed_names=()

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  exp="${m%.m}.expected"

  ir="$("$MATLABC" -emit-mlir -g "$m" 2>&1)"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    fail=$((fail+1))
    failed_names+=("$m (matlabc exit $rc)")
    echo "FAIL $m"
    printf '%s\n' "$ir" | sed 's/^/  /'
    continue
  fi

  got="$(printf '%s\n' "$ir" | python3 "$EXTRACT")"
  ext_rc=$?
  if [[ $ext_rc -ne 0 ]]; then
    fail=$((fail+1))
    failed_names+=("$m (extractor exit $ext_rc)")
    echo "FAIL $m (extractor)"
    continue
  fi

  if [[ -n "${UPDATE:-}" || ! -e "$exp" ]]; then
    printf '%s\n' "$got" > "$exp"
    echo "UPDATED $m"
    continue
  fi

  if ! diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    fail=$((fail+1))
    failed_names+=("$m (line mismatch)")
    echo "FAIL $m (line mismatch)"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
    continue
  fi

  if ! printf '%s\n' "$got" | python3 "$CHECK" "$m"; then
    fail=$((fail+1))
    failed_names+=("$m (non-executable hook line)")
    echo "FAIL $m (non-executable hook line)"
    continue
  fi

  pass=$((pass+1))
done

echo "----"
echo "passed: $pass    failed: $fail"
if [[ $fail -gt 0 ]]; then
  printf 'failures:\n'
  for n in "${failed_names[@]}"; do printf '  %s\n' "$n"; done
fi
exit $(( fail > 0 ? 1 : 0 ))
