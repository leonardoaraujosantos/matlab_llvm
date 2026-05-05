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
  # Tier-5 — added after the Python-emit slot-triplet handler shipped.
  # `matlab.alloc / matlab.store / matlab.load` now collapse to plain
  # Python variable reads/writes via DirectSlots.
  crc8
  # Tier-6 — added after the saturation pass: persistent stores +
  # function returns wrap to declared widths via rt.fi_wrap_*, and
  # _eq tolerates sign-interpretation mismatch (compares modulo
  # 2^WL when both values are integer-typed). Three of the eight
  # original class-3 modules still fail because their SV does
  # mid-computation wrap on every i16 op — needs per-op wrap
  # rather than just at register / return boundaries.
  aes_round
  barrel_shifter
  crc32
  fnv1a
  multi_cycle_mul
  # Tier-7 — added after the per-op wrap pass: every overflow-capable
  # arith op (addi / subi / muli / shli) and the matlab.* binop
  # equivalents wrap to their declared bit width, mirroring SV's
  # mid-computation truncation. Also fixes shrui to mask the
  # operand to its unsigned bit pattern before shifting (Python's
  # `>>` is arithmetic, SV's `>>` is logical). PersistWrapSpec
  # prefers the init's declared width over later body-side
  # bit-growth attrs.
  cordic_pipe
  cordic_step
)

# Run each fixture in parallel via xargs -P. Default to up to 8
# workers (about the right shape for a typical laptop / CI runner);
# override with COCOTB_PARALLEL=N. Each worker writes a per-fixture
# status line to a result file; the summary collates them after the
# parallel block finishes so the output stays deterministic
# regardless of which fixture finishes first.
PARALLEL="${COCOTB_PARALLEL:-8}"
RESULTS="$WORK_DIR/_results.txt"
: > "$RESULTS"

run_one() {
  local m=$1
  local out_dir="$WORK_DIR/$m"
  if ! "$MATLABC" -emit-cocotb \
                  -cocotb-out="$out_dir" \
                  "$ROOT/examples/hdl/$m.m" \
                  > "$out_dir.emit.log" 2>&1; then
    echo "$m EMIT-FAIL" >> "$RESULTS"
    return
  fi
  if (cd "$out_dir" && make > "$out_dir.run.log" 2>&1); then
    local ok
    ok=$(grep -oE "TESTS=1 PASS=[0-9]+ FAIL=[0-9]+" "$out_dir.run.log" | head -1)
    if [[ "$ok" == "TESTS=1 PASS=1 FAIL=0" ]]; then
      echo "$m PASS" >> "$RESULTS"
    else
      echo "$m FAIL ($ok)" >> "$RESULTS"
    fi
  else
    echo "$m MAKE-FAIL" >> "$RESULTS"
  fi
}

export -f run_one
export MATLABC ROOT WORK_DIR RESULTS

printf '%s\n' "${CASES[@]}" | xargs -n1 -P"$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

# Reorder the results into the same order as CASES for stable output
# regardless of completion order, then summarise.
pass=0; fail=0
for m in "${CASES[@]}"; do
  line=$(grep -m1 "^$m " "$RESULTS" || echo "$m UNKNOWN")
  status=${line#* }
  printf "  %-26s %s\n" "$m" "$status"
  case "$status" in
    PASS) pass=$((pass+1)) ;;
    *) fail=$((fail+1))
       # Print failure tail under the entry for easier triage.
       if [[ -f "$WORK_DIR/$m.run.log" ]]; then
         tail -10 "$WORK_DIR/$m.run.log" | sed 's/^/    /'
       elif [[ -f "$WORK_DIR/$m.emit.log" ]]; then
         cat "$WORK_DIR/$m.emit.log" | sed 's/^/    /'
       fi
       ;;
  esac
done

echo "----"
echo "cocotb-tests: $pass passed, $fail failed (parallel=$PARALLEL)"
[[ $fail -eq 0 ]]
