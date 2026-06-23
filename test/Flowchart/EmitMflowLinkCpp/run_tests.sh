#!/usr/bin/env bash
# Tier-G smoke test for `-emit-mflowlink-cpp`.
#
# For each fixture .mflow, emit the C++, compile via
# runtime/scripts/build_mflowlink.sh, run the binary, and diff its CSV
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
EX="$ROOT/examples/mflowlink"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# #50 Phase 5 — inline the former runtime/scripts/build_mflowlink.sh
# script.  The mflowlink C++ output links against the Flowchart static
# libs already produced by the CMake build (next to matlabc).  No
# wrapper shell script needed.
LIB="$(dirname "$MATLABC")"
CXX="${CXX:-$(command -v clang++ || command -v c++)}"
build_one_cpp() {
  local src="$1" out="$2"
  "$CXX" -std=c++17 -O2 -I "$ROOT/include" "$src" \
      "$LIB/libMatlabFlowchart.a" "$LIB/libMatlabParse.a" \
      "$LIB/libMatlabLex.a" "$LIB/libMatlabAST.a" \
      "$LIB/libMatlabBasic.a" -o "$out"
}

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
  if ! build_one_cpp "$cpp" "$bin" > "$SCRATCH/build.log" 2>&1; then
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

# The compiled C++ codegen must reproduce the in-process interpreter
# byte-for-byte. Every fixture below has been verified to round-trip
# (emit → compile → run → diff). Excluded: `matlab_fcn_jit` — a
# JIT-path-specific fixture; the C++ codegen lowers a MATLAB Function
# block's body through the AST interpreter (not the JIT), so it does not
# reproduce a JIT-only reference (same family as the JIT/AST divergence
# tracked in #77).
for name in lowpass pid_tracking multirate saturation_zc enabled_subsystem \
            thermostat tier_h_showcase tier_h_logic \
            goto_from matlab_fcn vector_signals sample_time_inherit \
            algebraic_loop_solved masked_library matlab_function_block \
            discrete_pid discrete_integrator_methods bus_signals \
            bouncing_ball stiff_bdf fir_filter per_flow_solver \
            matlab_fcn_loops matrix_signals \
            freefall_floor pid_block tier_h_discrete triggered_counter \
            state_space_vector_ic matlab_fcn_multi_output \
            awgn_channel error_rate psk_qpsk qam16 qpsk_awgn_link \
            biquad_lowpass streaming_filters fft_roundtrip window_taper \
            spectrum dwt_haar kalman_constant kalman_tracker lqr_feedback \
            running_stats dnn_predict rl_agent rf_2port pose_transform \
            color_space image_blocks nd_reshape nd_color_image \
            hdl_registers hdl_jk_sr hdl_memory hdl_half_adder \
            hdl_full_adder hdl_shift_register hdl_freq_divider \
            workspace_io color_image_filter nd_permute nd_squeeze \
            ode23_decay; do
  run_one "$name"
done

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${failed_names[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
