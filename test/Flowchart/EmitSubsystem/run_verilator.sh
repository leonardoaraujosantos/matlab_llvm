#!/usr/bin/env bash
# Tier-5h — Verilator lint gating for the Embedded Coder SV emit.
#
# For every `.mflow` fixture under `examples/mflowlink/coder/`,
# emits the SV via `matlabc -emit-sv --subsystem <name>` and runs
# `verilator --lint-only` over the result.  Catches regressions
# the structural smoke tests can't — e.g. undriven nets, latch
# inference, syntactically-malformed SV that yosys synth tolerates
# but real Verilog parsers reject.
#
# Skips cleanly with a warning if `verilator` isn't on PATH so the
# CTest lane stays safe to enable on machines that don't have it.
#
# Usage: run_verilator.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

if ! command -v verilator >/dev/null 2>&1; then
  echo "warning: verilator not found on PATH — skipping SV lint lane"
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/mflowlink/coder"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0; fail=0; skip=0
fails=()

# Subsystems known to emit clean SV (i.e. exercised by the
# structural smoke tests in run_tests.sh).  Each entry is
# `<.mflow stem>:<subsystem name>`.
declare -a SV_FIXTURES=(
  "scaled_sum_sv:scaled_sum"
  "stateless_mixer:stateless_mixer"
  "tapped_delay:tapped_delay"
  "fir_4tap:fir_4tap"
  "discrete_pid:discrete_pid"
  "continuous_lowpass:continuous_lowpass"
  "tf_lowpass:tf_lowpass"
  "tf_2nd_order:tf_2nd_order"
  "zp_plant:zp_plant"
  "transport_delay:transport_delay"
  "ss_plant:ss_plant"
  "mimo_state_space:mimo_state_space"
  "nested_pid_filter:outer_loop"
)

for spec in "${SV_FIXTURES[@]}"; do
  IFS=':' read -r stem name <<< "$spec"
  mflow="$EX/$stem.mflow"
  if [[ ! -f "$mflow" ]]; then
    skip=$((skip+1))
    continue
  fi
  sv="$SCRATCH/$stem.sv"
  if ! "$MATLABC" -emit-sv "$mflow" --subsystem "$name" \
       > "$sv" 2> "$SCRATCH/emit.err"; then
    fail=$((fail+1))
    fails+=("$stem (emit failure)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    continue
  fi
  # Verilator lint flags:
  # - -Wno-UNUSED:        we don't drive every reset path in always_comb
  # - -Wno-DECLFILENAME:  module name doesn't have to match the .sv file
  # - -Wno-ALWCOMBORDER:  the SV pipeline emits cosmetic
  #                       self-assignments (`d1 = d1;`) for single-slot
  #                       stateful blocks; benign at synth time but
  #                       Verilator flags the variable-driven-after-use.
  # - -Wno-UNOPTFLAT:     same root cause; Verilator can't statically
  #                       prove the combinational loop is broken.
  # - -Wno-WIDTHEXPAND:   the fi-saturate path widens i32 → i64 for
  #                       accumulator chains, then assigns the i64
  #                       sum back to an i32 output reg (wrap-on-
  #                       overflow matches simulator fi semantics).
  if ! verilator --lint-only -Wall \
       -Wno-UNUSED -Wno-DECLFILENAME \
       -Wno-ALWCOMBORDER -Wno-UNOPTFLAT \
       -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
       "$sv" > "$SCRATCH/lint.log" 2>&1; then
    fail=$((fail+1))
    fails+=("$stem (verilator lint)")
    head -20 "$SCRATCH/lint.log" | sed 's/^/  /' >&2
    continue
  fi
  pass=$((pass+1))
done

echo "----"
echo "passed: $pass    failed: $fail    skipped: $skip"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
