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

# Module list. Per-fixture latency lives in the source via
# `% cocotb: latency(N)` pragmas — the runner just walks every
# entry here without needing a hardcoded latency value. Modules
# blocked by Python-emit gaps (matlab.alloc / matlab.call_builtin
# handlers, SV-vs-Python fi saturation divergence, float-vs-int
# bitwise typing) are tracked separately — see
# docs/emit_cocotb.md "Python-emit gaps" — and not in this sweep.
declare -a CASES=(
  # Tier-1 — verified since v3.5 ship.
  alu_16bit
  counter_0_to_10
  fir_asic_pipelined        # `% cocotb: latency(4)` in source
  mealy_fsm
  moore_fsm
  mux_4to_1_16bit
  vector_processor
  sequential_processor      # `% cocotb: latency(4)` in source
  # Tier-2 — added after the 39-module sweep classified each
  # against the cocotb harness (random vectors at L=0). Every
  # module here passed without fixture-side tuning.
  computed_state_fsm
  hamming74
  i2c_bit_bang
  leading_zero_detector
  median3
  mmap_periph
  popcount
  priority_encoder
  pwm
  regfile
  rr_arbiter
  spi_master
  uart_rx
  up_down_counter
  # Tier-3 — added after the Python-emit `matlab.not` handler shipped.
  axi_handshake
  booth_mul
  edge_detector
  fifo
  manchester_enc
  sync_2ff                  # `% cocotb: latency(1)` in source — see B1
  # Tier-4 — added after the Python-emit persistent-init recognizer
  # learned the `isempty(p) || reset` shape. Capturing the in-body
  # init expr makes the module-level decl carry the right value
  # (e.g. `state = 1` instead of the `0.0` fallback), which fixes
  # downstream bitwise-on-float TypeErrors.
  async_fifo
  galois_lfsr
)

pass=0; fail=0
for m in "${CASES[@]}"; do
  out_dir="$WORK_DIR/$m"
  printf "  %-26s " "$m"
  if ! "$MATLABC" -emit-cocotb \
                  -cocotb-out="$out_dir" \
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
