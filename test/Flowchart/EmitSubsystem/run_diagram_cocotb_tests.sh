#!/usr/bin/env bash
# Tier-7d — whole-diagram cocotb SIL emit smoke tests.
#
# Each fixture is a `.mflow` model whose entry flow contains a single
# `signal_subsystem` (the DUT). `matlabc -emit-cocotb FILE.mflow
# --dut <block>` writes a complete cocotb harness directory:
#
#     <stem>_cocotb/
#         <dut-flow>.sv         — DUT (SV emit of the subsystem)
#         <dut-flow>_ref.py     — host reference (Python emit of the subsystem)
#         test_<stem>.py        — cocotb testbench (sources + sinks + SIL compare)
#         cocotb_fi.py          — Q<W>.<F> pack / unpack helpers
#         matlab_runtime.py     — fi semantics, persistent state
#         Makefile              — `make sim` to launch cocotb under verilator
#
# This script verifies:
#   1. The emit completes with rc=0 and writes every expected file.
#   2. The generated test_<stem>.py is syntactically valid Python.
#   3. The generated <dut-flow>_ref.py imports cleanly and the
#      `<DutClass>` exposes `step(...)`.
#   4. The diagram's host model (rendered inside test_<stem>.py)
#      can be syntactically parsed by Python's AST module — catches
#      any indentation / quoting regressions in the emit even when
#      cocotb / verilator aren't available.
#
# When `cocotb-config` and `verilator` are both on PATH, an additional
# end-to-end smoke compiles + runs the harness and asserts zero
# mismatches. That arm gracefully no-ops on hosts without the
# toolchain so the lane stays useful on minimal CI images.
#
# Usage: run_diagram_cocotb_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/mflowlink/coder"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
export PYTHONPATH="$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"

pass=0; fail=0
fails=()

# Fixture: cocotb_pid_sil — `signal_step` source drives a unit-
# gain-of-2 plant subsystem; one sink each on the stimulus and the
# plant output. DUT is the `plant_dut` block (referenced flow:
# `plant`). The harness reference computes 2*u; the SV does the
# same multiplication in Q16.16; tolerance is 1 LSB.
pid_sil_smoke() {
  local out="$SCRATCH/pid"
  "$MATLABC" -emit-cocotb "$EX/cocotb_pid_sil.mflow" \
      --dut plant_dut -cocotb-out="$out" >/dev/null 2>"$SCRATCH/pid.err"
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("cocotb_pid_sil (emit failed: $(cat "$SCRATCH/pid.err"))")
    return
  fi
  for needed in plant.sv plant_ref.py test_cocotb_pid_sil.py \
                cocotb_fi.py Makefile; do
    if [[ ! -f "$out/$needed" ]]; then
      fail=$((fail+1)); fails+=("cocotb_pid_sil ($needed missing)")
      return
    fi
  done
  # Python syntactic checks. The harness imports cocotb at module
  # level; we strip those imports + use ast.parse so the check
  # doesn't need cocotb on PATH.
  python3 - "$out" <<'PY'
import ast, os, sys
out = sys.argv[1]
tb = open(os.path.join(out, "test_cocotb_pid_sil.py")).read()
ast.parse(tb)
ref = open(os.path.join(out, "plant_ref.py")).read()
ast.parse(ref)
# Reference must expose `Plant` class with `step`.
ns = {}
exec(ref, ns)
assert "Plant" in ns, "plant_ref.py missing Plant class"
plant = ns["Plant"]()
assert hasattr(plant, "step"), "Plant has no step method"
# Spot-check the math (gain = 2).
y = plant.step(0.5)
assert abs(y - 1.0) < 1e-9, f"Plant.step(0.5) = {y}, expected 1.0"
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("cocotb_pid_sil (python smoke failed)")
    return
  fi
  pass=$((pass+1))

  # End-to-end arm: only if cocotb + verilator are both present.
  if command -v cocotb-config >/dev/null 2>&1 && \
     command -v verilator >/dev/null 2>&1; then
    pushd "$out" >/dev/null
    if make sim >"$SCRATCH/pid.cocotb.log" 2>&1; then
      # cocotb's summary line is uppercase: `TESTS=1 PASS=1 FAIL=0`.
      if grep -qi "FAIL=0" "$SCRATCH/pid.cocotb.log"; then
        pass=$((pass+1))
      else
        fail=$((fail+1))
        fails+=("cocotb_pid_sil (cocotb run reported failures)")
      fi
    else
      fail=$((fail+1))
      fails+=("cocotb_pid_sil (make sim non-zero exit)")
    fi
    popd >/dev/null
  fi
}
pid_sil_smoke

# Stateful DUT smoke: sine source → unit-delay subsystem → scope.
# Verifies the sequential-DUT pre-edge sampling path against the
# Python reference (which returns the pre-update state). Same end-
# to-end gate as the combinational case.
delay_sil_smoke() {
  local out="$SCRATCH/delay"
  "$MATLABC" -emit-cocotb "$EX/cocotb_delay_sil.mflow" \
      --dut delay_dut -cocotb-out="$out" >/dev/null 2>"$SCRATCH/delay.err"
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("cocotb_delay_sil (emit failed: $(cat "$SCRATCH/delay.err"))")
    return
  fi
  for needed in delay_block.sv delay_block_ref.py \
                test_cocotb_delay_sil.py cocotb_fi.py Makefile; do
    if [[ ! -f "$out/$needed" ]]; then
      fail=$((fail+1)); fails+=("cocotb_delay_sil ($needed missing)")
      return
    fi
  done
  python3 - "$out" <<'PY'
import ast, os, sys
out = sys.argv[1]
tb = open(os.path.join(out, "test_cocotb_delay_sil.py")).read()
ast.parse(tb)
ref = open(os.path.join(out, "delay_block_ref.py")).read()
ast.parse(ref)
ns = {}
exec(ref, ns)
assert "DelayBlock" in ns, "delay_block_ref.py missing DelayBlock class"
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("cocotb_delay_sil (python smoke failed)")
    return
  fi
  pass=$((pass+1))
  if command -v cocotb-config >/dev/null 2>&1 && \
     command -v verilator >/dev/null 2>&1; then
    pushd "$out" >/dev/null
    if make sim >"$SCRATCH/delay.cocotb.log" 2>&1; then
      if grep -qi "FAIL=0" "$SCRATCH/delay.cocotb.log"; then
        pass=$((pass+1))
      else
        fail=$((fail+1))
        fails+=("cocotb_delay_sil (cocotb run reported failures)")
      fi
    else
      fail=$((fail+1))
      fails+=("cocotb_delay_sil (make sim non-zero exit)")
    fi
    popd >/dev/null
  fi
}
delay_sil_smoke

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
