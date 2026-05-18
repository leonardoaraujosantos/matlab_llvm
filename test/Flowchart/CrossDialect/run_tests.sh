#!/usr/bin/env bash
# §17.5 #6 smoke test: compile a control-flow MATLAB script that
# invokes `mflowlink_run`, run it, and confirm the output has the
# expected shape (a row of doubles from the .mflow's logged signals).
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0
fail=0

# Copy the .m + .mflow into scratch so the script's relative path
# resolves regardless of where this test runs from.
cp "$ROOT/examples/mflowlink/cross_dialect.m" "$SCRATCH/"
mkdir -p "$SCRATCH/examples/mflowlink"
cp "$ROOT/examples/mflowlink/lowpass.mflow" \
   "$SCRATCH/examples/mflowlink/"

cd "$SCRATCH"

if ! "$ROOT/runtime/scripts/build_and_run.sh" "$SCRATCH/cross_dialect.m" \
     "$SCRATCH/cd" > "$SCRATCH/build.log" 2>&1; then
  echo "FAIL build_and_run.sh exit non-zero"
  sed 's/^/  /' "$SCRATCH/build.log" | tail -20
  exit 1
fi

OUT=$("$SCRATCH/cd")
echo "--- program output ---"
echo "$OUT"
echo "----------------------"

if grep -q 'lowpass logged-signal final values' <<<"$OUT"; then
  pass=$((pass+1))
  echo "PASS  output banner present"
else
  fail=$((fail+1)); echo "FAIL  output banner missing"
fi

# The lowpass settles around |0.314| amplitude. At t=10 (sine = 0),
# the scope is somewhere near zero with the LPF's residual ~-0.31.
# We accept any negative value with magnitude in [0.2, 0.4] in the
# second column (the `scope` block).
SCOPE_VAL=$(echo "$OUT" | awk '/lowpass logged-signal/ {getline; print $2}')
if awk -v v="$SCOPE_VAL" 'BEGIN { exit !(v <= -0.2 && v >= -0.4) }'; then
  pass=$((pass+1))
  echo "PASS  scope value ≈ -0.31 ($SCOPE_VAL)"
else
  fail=$((fail+1)); echo "FAIL  scope value out of range: $SCOPE_VAL"
fi

echo "----"
echo "passed: $pass  failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
