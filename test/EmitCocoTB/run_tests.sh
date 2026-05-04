#!/usr/bin/env bash
# CocoTB verification sweep across every `examples/hdl/*.m` module
# whose status is "supported" (see docs/emit_cocotb.md). Drives the
# generated SV DUT and the Python reference in lockstep with random
# (or pragma-driven) stimulus; asserts each module reports
# `TESTS=1 PASS=1 FAIL=0` from cocotb.
#
# Skip-if-missing policy: returns CTest's SKIP_RETURN_CODE (77)
# when verilator or cocotb isn't available — matches the policy
# the emit-typescript lane uses for Node. Lets the lane stay green
# on dev machines without the optional HDL toolchain.
#
# Usage: run_tests.sh <path-to-matlabc>

set -u
SKIP_RC=77

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

if ! command -v verilator >/dev/null 2>&1; then
  echo "cocotb-tests: SKIP (verilator not on PATH)"
  exit $SKIP_RC
fi
if ! command -v cocotb-config >/dev/null 2>&1; then
  echo "cocotb-tests: SKIP (cocotb not installed)"
  exit $SKIP_RC
fi
if ! python3 -c "import cocotb" >/dev/null 2>&1; then
  echo "cocotb-tests: SKIP (cocotb python module unavailable)"
  exit $SKIP_RC
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="$(mktemp -d -t matlabc_cocotb.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

# (module, latency) pairs. The latency mirrors what
# `just test-cocotb` uses; mismatches here mean a fixture
# regression, not a missing toolchain.
#
# fir_asic_pipelined runs at latency=4 with gain held constant
# (via `% cocotb: stimulus(gain, constant, 0.25)` in the source).
# The 4-cycle latency reflects the actual pipeline depth:
# delay_line → reg_products → reg_acc → reg_output. Holding gain
# constant ensures the SV's per-cycle gain×reg_acc multiplication
# matches the Python reference's single-call full-pipe computation
# (otherwise the SV multiplies x0's accumulator by gain at cycle
# 3, while Python pairs them at the same call).
declare -a CASES=(
  "alu_16bit:0"
  "counter_0_to_10:0"
  "fir_asic_pipelined:4"
  "mealy_fsm:0"
  "moore_fsm:0"
  "mux_4to_1_16bit:0"
  "vector_processor:0"
  "sequential_processor:4"
)

pass=0; fail=0
for entry in "${CASES[@]}"; do
  m=${entry%%:*}
  L=${entry##*:}
  out_dir="$WORK_DIR/$m"
  printf "  %-26s L=%s  " "$m" "$L"
  if ! "$MATLABC" -emit-cocotb \
                  -cocotb-out="$out_dir" \
                  -cocotb-latency=$L \
                  "$ROOT/examples/hdl/$m.m" \
                  > "$out_dir.emit.log" 2>&1; then
    echo "EMIT FAIL"
    cat "$out_dir.emit.log" | sed 's/^/    /'
    fail=$((fail+1))
    continue
  fi
  if (cd "$out_dir" && make > "$out_dir.run.log" 2>&1); then
    ok=$(grep -oE "TESTS=1 PASS=[0-9]+ FAIL=[0-9]+" "$out_dir.run.log" | head -1)
    if [[ "$ok" == "TESTS=1 PASS=1 FAIL=0" ]]; then
      echo "PASS"
      pass=$((pass+1))
    else
      echo "FAIL ($ok)"
      tail -20 "$out_dir.run.log" | sed 's/^/    /'
      fail=$((fail+1))
    fi
  else
    echo "MAKE FAIL"
    tail -20 "$out_dir.run.log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
done

echo "----"
echo "cocotb-tests: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
