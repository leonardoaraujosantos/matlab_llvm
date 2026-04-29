#!/usr/bin/env bash
# Phase 2 golden-file runner: each *.mflow under this dir is fed
# through `matlabc -emit-matlab` and the combined stdout+stderr is
# diffed against the matching *.expected file. Files under Errors/
# are expected to fail (the golden captures the diagnostic output).
#
# Usage: run_tests.sh <path-to-matlabc>
#        UPDATE=1 run_tests.sh <path-to-matlabc>
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

run_one() {
  local f="$1"
  local exp="${f%.mflow}.expected"
  local got
  # Phase 4b: a `library_id` custom block needs `--block-path` to
  # resolve. Wire the local `lib/` for any fixture that mentions
  # "library_id"; non-library tests get an empty extra-flag list.
  local flags=()
  if grep -q '"library_id"' "$f"; then
    flags=(--block-path "$TESTDIR/lib")
  fi
  got="$("$MATLABC" ${flags[@]+"${flags[@]}"} -emit-matlab "$f" 2>&1 | sed "s|$TESTDIR/||g")" || true
  if [[ -n "${UPDATE:-}" || ! -e "$exp" ]]; then
    printf '%s\n' "$got" > "$exp"
    echo "UPDATED $f"
    return
  fi
  if diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $f"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
}

shopt -s nullglob
for f in *.mflow Errors/*.mflow; do
  run_one "$f"
done

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
