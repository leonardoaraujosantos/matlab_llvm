#!/usr/bin/env bash
# Tier-5j — Verilator behavioural cosim for the Embedded Coder SV
# emit. For each fixture in the curated PASSING list, drives the
# same stimulus through (a) the Verilator-compiled SV and (b) the
# Python emit's class wrapper, then compares per-tick outputs with
# tolerance.
#
# Skips cleanly (exit 0 + warning) when verilator isn't on PATH so
# the surrounding CTest lane is safe to enable everywhere.
#
# Why a curated list rather than every SV-capable fixture: the
# matlab_llvm SV emit currently has two pre-existing bugs that this
# cosim correctly identifies, but doesn't fix:
#
#   - **fi-multiplication missing the Q<W>.<F> normalising shift.**
#     `fi(K, 1, 32, 16) .* x` lowers to `(x << log2(K_raw))` without
#     the trailing `>>> 16`, so the high bits truncate when the
#     wider intermediate gets stored into a 32-bit register. Visible
#     in any fixture with a Gain block, a Sum-of-products, a TF, a
#     ZP, or a MIMO/SISO state-space block — i.e. most of the
#     interesting fixtures. Fix: thread `fi.cast` ops around each
#     fi-multiplication's result so LowerFixedPoint inserts the
#     right-shift. Carved out in docs/embedded_coder_roadmap.md
#     (Tier-5j).
#
#   - **Stateful blocks' state-read hoist emits `local = local`
#     (self-assignment) for the multi-output `tapped_delay`-style
#     pattern.** The state-read should write `local = state_reg` but
#     the SV pipeline lowers it to the cosmetic self-assign that
#     Verilator's UNOPTFLAT warning flags (suppressed today). The
#     local then reads uninitialised. Fix: separate diagnosis;
#     likely a slot-promotion ordering issue.
#
# The four currently-passing fixtures exercise (i) pure stateless
# combinational logic with no fi-multiplication, (ii) pure-delay
# state machines (Unit Delay / transport delay) where the output IS
# the state register read.
#
# Usage: run_verilator_cosim.sh <path-to-matlabc>

set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

if ! command -v verilator >/dev/null 2>&1; then
  echo "warning: verilator not found on PATH — skipping SV cosim lane"
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/mflowlink/coder"
COSIM="$ROOT/test/Flowchart/EmitSubsystem/cosim.py"

pass=0; fail=0
fails=()

# Each entry: <fixture-stem>:<subsystem-name>:<stimulus>:<ticks>:<tolerance>
declare -a CASES=(
  # Pure-delay state machines — state read IS the output, no fi
  # multiplication in the data path. Bit-exact agreement expected.
  "unit_delay:unit_delay:step_first:30:0.001"
  "transport_delay:transport_delay:step:30:0.001"

  # Stateless combinational with boolean outputs. The cosim
  # decodes 1-bit output ports as boolean rather than Q16.16.
  "comparator_logic:comparator_logic:step_first:30:0.001"
  "threshold_switch:threshold_switch:step_first:30:0.001"

  # Tier-5k bug fixes: every fixture with fi-multiplication or
  # multi-output stateful blocks now passes within Q16.16 noise.
  "stateless_mixer:stateless_mixer:step_first:30:0.001"
  "tapped_delay:tapped_delay:step:30:0.001"
  "tf_lowpass:tf_lowpass:step_first:30:0.005"
  "fir_4tap:fir_4tap:step_first:30:0.001"
  "discrete_pid:discrete_pid:step_first:30:0.005"
  "mimo_state_space:mimo_state_space:step_first:30:0.005"
  "tf_2nd_order:tf_2nd_order:step_first:30:0.005"
  "zp_plant:zp_plant:step_first:30:0.005"
  "ss_plant:ss_plant:step_first:30:0.005"
  "continuous_lowpass:continuous_lowpass:step_first:30:0.005"
  # Tier-6 — nested subsystem (outer + inner helper modules).
  "nested_pid_filter:outer_loop:step_first:30:0.005"
  # Tier-6b — signal_matlab_fcn user-MATLAB body wrapped at args.
  "matlab_fcn_sv:matlab_fcn_sv:step_first:20:0.005"
  # Tier-6c — multirate (fast + slow Unit Delay, per-block phase
  # counters). Same level-change stimulus as the software smoke
  # test; cosim verifies the SV emit's HDL multirate gating matches
  # the Python emit's software multirate gating bit-for-bit.
  "multirate_filters:multirate_filters:step_first:30:0.001"
)

for spec in "${CASES[@]}"; do
  IFS=':' read -r stem name stim ticks tol <<< "$spec"
  mflow="$EX/$stem.mflow"
  if [[ ! -f "$mflow" ]]; then
    fail=$((fail+1)); fails+=("$stem (missing fixture)")
    continue
  fi
  if python3 "$COSIM" "$MATLABC" "$mflow" "$name" \
       --ticks "$ticks" --tol "$tol" --stimulus "$stim" 2>&1 | tail -1; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); fails+=("$name (cosim mismatch)")
  fi
done

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
