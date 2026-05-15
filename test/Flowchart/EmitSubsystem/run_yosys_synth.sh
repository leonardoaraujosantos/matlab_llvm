#!/usr/bin/env bash
# Tier-5l — yosys synthesis gate over every SV-capable fixture.
#
# Catches what neither the structural lint nor the behavioural cosim
# catches: latch inference, unsynthesisable constructs, ungrouped
# state, and the per-fixture gate-count sanity check (a wildly
# different count from one PR to the next is usually a signal of an
# emit regression). Runs yosys's generic `synth` pass — no target
# library, no real fmax — for portability across machines.
#
# Skips cleanly (exit 0 + warning) when `yosys` isn't on PATH so the
# surrounding CTest lane is safe to enable everywhere.
#
# Usage: run_yosys_synth.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

if ! command -v yosys >/dev/null 2>&1; then
  echo "warning: yosys not found on PATH — skipping SV synth lane"
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/mflowlink/coder"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0; fail=0
fails=()

# Each entry: `<.mflow stem>:<subsystem-name>:<min-cells>`.  The
# minimum-cells bound is a sanity floor (yosys returns 0 cells for
# trivially-DCE'd modules); the upper bound is implicitly the
# fixture's complexity. A wildly different gate count from one PR to
# the next is usually a signal of an emit regression — re-run the
# lane with `-v` to see the per-fixture stat dump.
declare -a CASES=(
  "stateless_mixer:stateless_mixer:50"
  "comparator_logic:comparator_logic:1"
  "threshold_switch:threshold_switch:20"
  "scaled_sum_sv:scaled_sum:50"
  # matlab_fcn_sv carved out — pre-existing SV emit limitation
  # where the user `signal_matlab_fcn` body's return type isn't
  # fi-inferred, so the outport fi-wrap routes through the
  # non-synthesisable matlab_fi_quantize_s constructor. Fix path:
  # propagate operand fi specs through the user-fn body's binops.
  "unit_delay:unit_delay:30"
  "tapped_delay:tapped_delay:80"
  "transport_delay:transport_delay:80"
  "fir_4tap:fir_4tap:200"
  "discrete_pid:discrete_pid:500"
  "continuous_lowpass:continuous_lowpass:200"
  "tf_lowpass:tf_lowpass:200"
  "tf_2nd_order:tf_2nd_order:500"
  "zp_plant:zp_plant:500"
  "ss_plant:ss_plant:500"
  "mimo_state_space:mimo_state_space:200"
)

for spec in "${CASES[@]}"; do
  IFS=':' read -r stem name min_cells <<< "$spec"
  mflow="$EX/$stem.mflow"
  if [[ ! -f "$mflow" ]]; then
    fail=$((fail+1)); fails+=("$stem (missing fixture)")
    continue
  fi
  sv="$SCRATCH/$stem.sv"
  if ! "$MATLABC" -emit-sv "$mflow" --subsystem "$name" \
       > "$sv" 2> "$SCRATCH/emit.err"; then
    fail=$((fail+1)); fails+=("$stem (emit failure)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    continue
  fi
  # `synth` runs the generic flatten + opt + memory + abc/techmap
  # cascade. We pipe through grep to extract just the cell count
  # without dragging the full log into the CTest output.
  if ! yosys -p "read_verilog -sv $sv; synth -top $name; stat" \
       > "$SCRATCH/synth.out" 2> "$SCRATCH/synth.err"; then
    fail=$((fail+1)); fails+=("$stem (yosys synth)")
    tail -20 "$SCRATCH/synth.err" | sed 's/^/  /' >&2
    continue
  fi
  # The `stat` pass prints a "<N> cells" line per design. The
  # top-level design's count comes last (after submodule stats).
  cells=$(grep -E "^ +[0-9]+ cells$" "$SCRATCH/synth.out" \
          | tail -1 | awk '{print $1}')
  if [[ -z "$cells" ]]; then
    fail=$((fail+1)); fails+=("$stem (stat parse)")
    continue
  fi
  if (( cells < min_cells )); then
    fail=$((fail+1)); fails+=("$stem ($cells cells < min $min_cells)")
    continue
  fi
  if [[ "${VERBOSE:-0}" != "0" ]]; then
    printf "  %-22s %d cells\n" "$stem" "$cells"
  fi
  pass=$((pass+1))
done

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
