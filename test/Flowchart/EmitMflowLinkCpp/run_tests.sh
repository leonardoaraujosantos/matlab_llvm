#!/usr/bin/env bash
# Tier-G smoke test for `-emit-mflowlink-cpp`.
#
# For each fixture .mflow, emit the C++, compile via
# runtime/build_mflowlink.sh, run the binary, and diff its CSV
# against `matlabc -simulate` (the in-process interpreter) on the
# same input. A byte-identical match means the standalone codegen
# lane reproduces the simulation exactly.
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
BUILDER="$ROOT/runtime/build_mflowlink.sh"
EX="$ROOT/examples/mflowlink"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0
fail=0
failed_names=()

run_one() {
  local name="$1"
  local in="$EX/$name.mflow"
  local cpp="$SCRATCH/$name.cpp"
  local bin="$SCRATCH/$name"
  local gen_csv="$SCRATCH/$name.gen.csv"
  local ref_csv="$SCRATCH/$name.ref.csv"

  if ! "$MATLABC" -emit-mflowlink-cpp "$in" > "$cpp" 2> "$SCRATCH/emit.err"; then
    fail=$((fail+1)); failed_names+=("$name (emit)")
    echo "FAIL $name (emit)"
    sed 's/^/  /' "$SCRATCH/emit.err"
    return
  fi
  if ! "$BUILDER" "$cpp" "$bin" > "$SCRATCH/build.log" 2>&1; then
    fail=$((fail+1)); failed_names+=("$name (build)")
    echo "FAIL $name (build)"
    sed 's/^/  /' "$SCRATCH/build.log"
    return
  fi
  "$bin" > "$gen_csv" 2> "$SCRATCH/run.err"
  "$MATLABC" -simulate "$in" > "$ref_csv" 2> /dev/null
  if ! diff -q "$ref_csv" "$gen_csv" > /dev/null; then
    fail=$((fail+1)); failed_names+=("$name (csv)")
    echo "FAIL $name (csv mismatch)"
    diff -u "$ref_csv" "$gen_csv" | head -20 | sed 's/^/  /'
    return
  fi
  pass=$((pass+1))
}

for name in lowpass pid_tracking multirate saturation_zc enabled_subsystem \
            thermostat tier_h_showcase tier_h_logic \
            goto_from matlab_fcn vector_signals sample_time_inherit \
            algebraic_loop_solved masked_library matlab_function_block \
            discrete_pid discrete_integrator_methods bus_signals \
            bouncing_ball stiff_bdf fir_filter per_flow_solver \
            matlab_fcn_loops matrix_signals; do
  run_one "$name"
done

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${failed_names[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
