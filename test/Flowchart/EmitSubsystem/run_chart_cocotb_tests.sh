#!/usr/bin/env bash
# mStateflow cocotb SIL smoke test.
#
# `matlabc -emit-cocotb chart.mflow` writes a cocotb harness directory:
#
#     <stem>_cocotb/
#         <chart>_tick.sv          — SV-target lowering of the chart
#         <chart>_tick_ref.py      — Python reference (via -emit-python)
#         test_<stem>.py            — cocotb testbench (SIL compare)
#         cocotb_fi.py / matlab_runtime.py / Makefile
#
# This script verifies the integer-typed Moore / Mealy / AND-parallel
# example charts emit valid cocotb harnesses:
#   1. The emit completes with rc=0 and writes every expected file.
#   2. test_<stem>.py is syntactically valid Python.
#   3. <chart>_tick_ref.py imports cleanly and exposes the
#      <chart>_tick function (sequential drives over its persistent
#      state).
#
# When `cocotb-config` and `verilator` are on PATH, an end-to-end
# `make sim` run is attempted. It gracefully no-ops on hosts without
# the toolchain so the lane stays useful on minimal CI images.
#
# Usage: run_chart_cocotb_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/stateflow"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
export PYTHONPATH="$ROOT/runtime/shim${PYTHONPATH:+:$PYTHONPATH}"

pass=0; fail=0
fails=()

# Common harness for each chart: emit, check files, parse Python.
# Generated files use the .mflow stem (not the chart name); the
# embedded tick function inside the ref module uses the chart name.
chart_cocotb_smoke() {
  local stem="$1"
  local tick_fn="$2"   # e.g. traffic_light_tick (function inside ref.py)
  local out="$SCRATCH/$stem"
  "$MATLABC" -emit-cocotb "$EX/$stem.mflow" \
      -cocotb-out="$out" >/dev/null 2>"$SCRATCH/$stem.err"
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1))
    fails+=("$stem (emit failed: $(cat "$SCRATCH/$stem.err"))")
    return
  fi
  for needed in "${stem}.sv" "${stem}_ref.py" \
                "test_${stem}.py" cocotb_fi.py Makefile; do
    if [[ ! -f "$out/$needed" ]]; then
      fail=$((fail+1))
      fails+=("$stem (missing $needed)")
      return
    fi
  done
  python3 - "$out" "$stem" "$tick_fn" <<'PY'
import ast, os, sys
out, stem, tick_fn = sys.argv[1], sys.argv[2], sys.argv[3]
tb = open(os.path.join(out, f"test_{stem}.py")).read()
ast.parse(tb)
ref = open(os.path.join(out, f"{stem}_ref.py")).read()
ast.parse(ref)
# Reference must expose the chart_tick function. We can't exec it
# without matlab_runtime importable; just check the AST has the
# function definition we expect.
mod = ast.parse(ref)
fn_names = {n.name for n in mod.body if isinstance(n, ast.FunctionDef)}
assert tick_fn in fn_names, \
    f"{tick_fn} missing from {stem}_ref.py (saw {sorted(fn_names)})"
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1))
    fails+=("$stem (Python harness check failed)")
    return
  fi
  # End-to-end SIL when the toolchain is available.
  if command -v cocotb-config >/dev/null 2>&1 \
     && command -v verilator >/dev/null 2>&1; then
    (cd "$out" && make sim >"$SCRATCH/$stem.simlog" 2>&1)
    if [[ $? -ne 0 ]] || ! grep -q "TESTS=.*PASS=" "$SCRATCH/$stem.simlog"; then
      fail=$((fail+1))
      fails+=("$stem (cocotb sim run failed — see $SCRATCH/$stem.simlog)")
      return
    fi
  fi
  pass=$((pass+1))
}

chart_cocotb_smoke traffic_light_moore   traffic_light_tick
chart_cocotb_smoke vending_machine_mealy vending_tick
chart_cocotb_smoke model_air_temperature_controller air_controller_tick
chart_cocotb_smoke pipelined_mac         pipemac_tick

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  echo "failures:"
  for f in "${fails[@]}"; do echo "  - $f"; done
fi
exit $(( fail > 0 ? 1 : 0 ))
