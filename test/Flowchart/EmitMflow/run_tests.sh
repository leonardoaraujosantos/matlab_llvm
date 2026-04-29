#!/usr/bin/env bash
# Phase 7 round-trip + idempotency test for `-emit-mflow`.
#
# For every input fixture under FIXTURES, the test runs:
#
#   1. `matlabc -emit-mflow input.<ext>` → first.mflow
#   2. `matlabc -emit-matlab first.mflow` → first.m
#   3. `matlabc -emit-mflow first.m`     → second.mflow
#
# It then diffs first.mflow against second.mflow byte-for-byte
# (idempotency from the second iteration onward — the canonical
# property a valid emitter must satisfy). It also confirms the
# generated MATLAB at step 2 parses cleanly.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
# Resolve to an absolute path before the upcoming `cd` shifts CWD.
case "$MATLABC" in
  /*) ;;
  *) MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")" ;;
esac

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$TESTDIR/../../.." && pwd)"
cd "$TESTDIR"

# Inputs: a hand-curated subset of examples/*.m and the existing
# .mflow corpus. The C-emitter limitation around if/else+break
# inside while makes while_loop.m / similar shapes round-trip to
# semantically-equivalent-but-textually-different forms — those
# fixtures are excluded here and exercised separately by the
# flowchart-cross-backend lane on .mflow inputs.
INPUTS=(
  "$REPO_ROOT/examples/hello.m"
  "$REPO_ROOT/examples/factorial.m"
  "$REPO_ROOT/examples/for_loop.m"
  "$REPO_ROOT/examples/is_old.m"
  "$REPO_ROOT/examples/matrix_mult.m"
  "$REPO_ROOT/examples/solve_linear.m"
  "$REPO_ROOT/examples/mflow/hello.mflow"
  "$REPO_ROOT/examples/mflow/factorial.mflow"
  "$REPO_ROOT/examples/mflow/for_loop.mflow"
  "$REPO_ROOT/examples/mflow/is_old.mflow"
  "$REPO_ROOT/examples/mflow/matrix_mult.mflow"
)

pass=0
fail=0

run_one() {
  local f="$1"
  local name; name=$(basename "$f")
  name="${name%.*}"
  local first="/tmp/emitmflow_${name}_1.mflow"
  local mid="/tmp/emitmflow_${name}_mid.m"
  local second="/tmp/emitmflow_${name}_2.mflow"

  if ! "$MATLABC" -emit-mflow "$f" > "$first" 2>/dev/null; then
    echo "FAIL $name (emit-mflow on input failed)"
    fail=$((fail+1))
    return
  fi
  if ! "$MATLABC" -emit-matlab "$first" > "$mid" 2>/dev/null; then
    echo "FAIL $name (round-tripped .mflow does not -emit-matlab)"
    fail=$((fail+1))
    return
  fi
  if ! "$MATLABC" -emit-mflow "$mid" > "$second" 2>/dev/null; then
    echo "FAIL $name (round-tripped .m does not -emit-mflow)"
    fail=$((fail+1))
    return
  fi
  if ! diff -u "$first" "$second" > /tmp/emitmflow_${name}.diff 2>&1; then
    echo "FAIL $name (idempotency broken: iter1 vs iter2)"
    head -20 /tmp/emitmflow_${name}.diff | sed 's/^/  /'
    fail=$((fail+1))
    return
  fi
  pass=$((pass+1))
}

for f in "${INPUTS[@]}"; do
  if [[ -e "$f" ]]; then
    run_one "$f"
  fi
done

echo "----"
echo "(idempotency cases)"
echo "passed: $pass    failed: $fail"

# Phase 8d: --preserve-layout merges `ui.position` from a reference
# file into the new emission for matching node ids. Build a v1 from
# a known input, hand-edit one block's position, re-emit with
# --preserve-layout, and assert the edited position survives in v2
# while unrelated nodes keep their auto-layout.
preserve_pass=0
preserve_fail=0
preserve_test() {
  local input="$REPO_ROOT/examples/factorial.m"
  local v1="/tmp/p8d_factorial_v1.mflow"
  local edited="/tmp/p8d_factorial_edited.mflow"
  local v2="/tmp/p8d_factorial_v2.mflow"

  "$MATLABC" -emit-mflow "$input" > "$v1" 2>/dev/null
  # Edit n_display_1's position in v1 to a sentinel (777, 999).
  python3 -c "
import json, sys
with open('$v1') as f: d = json.load(f)
hit = False
for fl in d['flows']:
    for n in fl['nodes']:
        if n.get('id') == 'n_display_1':
            n['ui']['position'] = {'x': 777, 'y': 999}
            hit = True
            break
    if hit: break
print(json.dumps(d, sort_keys=True, indent=2))
sys.exit(0 if hit else 1)
" > "$edited" || { echo "FAIL preserve-layout: couldn't find n_display_1 to edit"; preserve_fail=$((preserve_fail+1)); return; }

  # Re-emit with --preserve-layout.
  "$MATLABC" -emit-mflow --preserve-layout "$edited" "$input" > "$v2" 2>/dev/null

  # Verify n_display_1 still has the sentinel position; verify another
  # node (n_variable_1) does NOT have the sentinel (it should keep
  # auto-layout since the edited file didn't change its position).
  if python3 -c "
import json, sys
with open('$v2') as f: d = json.load(f)
display_pos = None
variable_pos = None
for fl in d['flows']:
    for n in fl['nodes']:
        if n.get('id') == 'n_display_1':
            display_pos = n['ui']['position']
        elif n.get('id') == 'n_variable_1':
            variable_pos = n['ui']['position']
ok = True
if display_pos != {'x': 777, 'y': 999}:
    print(f'n_display_1 sentinel not preserved: {display_pos}'); ok = False
if variable_pos == {'x': 777, 'y': 999}:
    print(f'n_variable_1 incorrectly received sentinel'); ok = False
sys.exit(0 if ok else 1)
"; then
    preserve_pass=$((preserve_pass+1))
  else
    echo "FAIL preserve-layout"
    preserve_fail=$((preserve_fail+1))
  fi
}
preserve_test
echo "(--preserve-layout cases)"
echo "passed: $preserve_pass    failed: $preserve_fail"

total_fail=$(( fail + preserve_fail ))
exit $(( total_fail > 0 ? 1 : 0 ))
