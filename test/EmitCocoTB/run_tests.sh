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
  sync_2ff                  # 2-flop synchronizer; latency=0 under snapshot-ref semantics
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
  # Tier-8 — added after the Python-emit pre-snapshot persistent-read
  # pass. Reads of persistents whose value flows to next-state writes
  # (not function outputs) route through `_<name>_snap` captured at
  # function entry, matching SV's always_comb non-blocking behaviour.
  # Closes the integrator-chain divergence that previously kept
  # cic_decimator out of the sweep.
  cic_decimator
  # Tier-9 — added with the `% cocotb: range(<port>, <lo>, <hi>)` pragma
  # that bounds the random stimulus to a real-value window.  Useful when
  # the natural fi_range would let stimulus exercise overflow regions
  # that SV's mid-computation truncation and the Python reference's
  # saturate-and-grow disagree on.  The fixture itself is a trivial
  # Q16.0 adder, so the cocotb compare is at the same FL on both sides
  # -- what gets exercised here is the new pragma + harness emit.
  cocotb_range_pragma
  # Tier-10 — DL HDL H3 bit-accuracy.  A Q16.8 quantized 2-2-1 MLP
  # forward using the `% hdl: precise_fi` pragma to enable Sema-mono
  # on the HW lane, so the FL-grown intermediates flow through both
  # the SV emit and the Python reference identically.  Closes the
  # documented bias-FL divergence (issue #75): SV bias now lowers as
  # `64'sd8192` (= 0.125 * 2^16) instead of the Q16.8 raw 32, matching
  # the Python ref's Q33.16 / Q34.16 internal precision.
  dlhdl_quant_mlp
  # Tier-11 — DL HDL H4 LSTM-on-FPGA.  Two precise_fi fixtures that
  # demonstrate the recurrent kernel compiles to bit-accurate fi SV:
  #   dlhdl_rnn_cell  — simple recurrent cell `h_new = hardtanh(Wx*x + Wh*h + b)`.
  #   dlhdl_lstm_cell — full LSTM cell with `hardsigmoid` (i/f/o gates) +
  #                     `hardtanh` (g gate + final hidden activation) +
  #                     cell-state update `c_new = f*c_prev + i*g`.
  # Both DUTs are combinational (caller threads h_prev / c_prev across
  # timesteps externally); the multi-timestep recurrence with persistent
  # registers is a documented follow-on slice.
  dlhdl_rnn_cell
  dlhdl_lstm_cell
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

# Per-fixture timeout. A hung verilator/cocotb run would otherwise
# pin one of the parallel workers indefinitely. Default 120s — well
# above the slowest fixture's normal runtime — overridable via
# COCOTB_FIXTURE_TIMEOUT. Falls back to no timeout if neither
# `timeout` nor `gtimeout` is on PATH (rare; both ship with GNU
# coreutils on Linux and via brew on macOS).
TIMEOUT_S="${COCOTB_FIXTURE_TIMEOUT:-120}"
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout $TIMEOUT_S"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout $TIMEOUT_S"
else
  TIMEOUT_CMD=""
  echo "warning: no timeout binary found; fixture runs are unbounded" >&2
fi

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
  # `timeout` exits 124 when the wrapped command was killed for
  # exceeding the limit. Distinguishing TIMEOUT from MAKE-FAIL in
  # the status output makes triage faster — a hung sim is a
  # different bug from a make-step error.
  local rc
  (cd "$out_dir" && $TIMEOUT_CMD make > "$out_dir.run.log" 2>&1)
  rc=$?
  if [[ $rc -eq 0 ]]; then
    local ok
    ok=$(grep -oE "TESTS=1 PASS=[0-9]+ FAIL=[0-9]+" "$out_dir.run.log" | head -1)
    if [[ "$ok" == "TESTS=1 PASS=1 FAIL=0" ]]; then
      echo "$m PASS" >> "$RESULTS"
    else
      echo "$m FAIL ($ok)" >> "$RESULTS"
    fi
  elif [[ $rc -eq 124 ]]; then
    echo "$m TIMEOUT (>${TIMEOUT_S}s)" >> "$RESULTS"
  else
    echo "$m MAKE-FAIL" >> "$RESULTS"
  fi
}

export -f run_one
export MATLABC ROOT WORK_DIR RESULTS TIMEOUT_CMD TIMEOUT_S

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
       # Replay hint — every harness drops args_trail.jsonl, so a
       # one-shell-line repro is always available regardless of
       # whether the failure was deterministic or a flake. The hint
       # is most useful when CI fails and the developer wants to
       # reproduce locally without re-running the whole sweep.
       if [[ -f "$WORK_DIR/$m/args_trail.jsonl" ]]; then
         echo "    repro: cd $WORK_DIR/$m && make replay"
       fi
       ;;
  esac
done

echo "----"
echo "cocotb-tests: $pass passed, $fail failed (parallel=$PARALLEL)"
[[ $fail -eq 0 ]]
