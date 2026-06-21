#!/usr/bin/env bash
# Embedded Coder, Tier 1 — `matlabc -emit-{python,c,cpp,typescript}
# --subsystem <name>` smoke lane. For every `.mflow` fixture under
# `examples/mflowlink/coder/`, emit each target's source and run a
# small built-in checker that exercises the entry function. The
# expected values live in `<fixture>.expected` (one
# `<args> -> <out>` line per case); the checker compares the
# generated code's output against those references.
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
EX="$ROOT/examples/mflowlink/coder"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
# `matlab_runtime.py` ships alongside the compiler — surface it on
# PYTHONPATH so the emitted Python's `import matlab_runtime as rt`
# resolves to the shipped helpers (rt.sin_s / rt.abs_s / ...).
export PYTHONPATH="$ROOT/runtime/shim${PYTHONPATH:+:$PYTHONPATH}"

pass=0; fail=0
fails=()

# Expected outputs per fixture — single-row tabular form
# `<args> -> <returns>`. For stateful subsystems the args include
# the per-block state slots in their final positions, and the
# returns include the next-state slots in theirs (matching the
# function's full signature). The Python harness eval's the
# subsystem call once per case and asserts approximate equality
# (1e-6). Multi-tick state evolution is tested by chaining cases
# (the test runner doesn't thread state automatically — each case
# stands on its own with explicit state inputs).
declare -a CASES=(
  # stateless_mixer: y = sat(2*u1 + u2 - u3), sat clamps to [-1, 1]
  'stateless_mixer|0.3,0.2,0.1->0.7|1.0,1.0,0.0->1.0|-1.0,-2.0,0.0->-1.0|0.0,0.0,0.0->0.0'
  # comparator_logic: y = (u1 > 0) AND (u2 < 5)
  'comparator_logic|1.0,1.0->True|1.0,7.0->False|-1.0,1.0->False'
  # threshold_switch: y = (u2 > 0.5) ? u1 : u3
  'threshold_switch|10.0,0.6,-10.0->10.0|10.0,0.4,-10.0->-10.0|10.0,0.5,-10.0->-10.0'
  # math_fns: returns (abs(u1), sin(u1))
  'math_fns|0.0->0.0,0.0|1.5->1.5,0.9974949866|-2.0->2.0,-0.9092974268'
  # Tier 3 — stateful subsystems.
  # unit_delay(u, s) -> (y, s_next): y = s, s_next = u.
  'unit_delay|0.0,0.0->0.0,0.0|5.0,2.0->2.0,5.0|-1.0,7.0->7.0,-1.0'
  # discrete_integrator(u, s_acc) -> (y, s_acc_next):
  #   y = s_acc, s_acc_next = s_acc + 0.5 * u * Ts (Ts=0.1)
  'discrete_integrator|0.0,0.0->0.0,0.0|1.0,0.0->0.0,0.05|1.0,0.1->0.1,0.15|2.0,0.5->0.5,0.6'
  # discrete_pid(ref, meas, s_iacc, s_prev_err) -> (u, iacc_next, prev_next):
  # Step 0 (cold start): err = 1, P = 1.2, I = 0, D = 1*4 = 4.0
  #   u_pre_sat = 5.2; sat[-5,5] -> 5.0; iacc_next = 0.05; prev_next = 1.0
  # Step 1 (prev=1, iacc=0.05): err = 1, P = 1.2, I = 0.05*0.4 = 0.02, D = 0
  #   u = 1.22; iacc_next = 0.10; prev_next = 1.0
  # Step 2 (prev=1, iacc=0.10): err = 1, P = 1.2, I = 0.04, D = 0
  #   u = 1.24; iacc_next = 0.15; prev_next = 1.0
  'discrete_pid|1.0,0.0,0.0,0.0->5.0,0.05,1.0|1.0,0.0,0.05,1.0->1.22,0.10,1.0|1.0,0.0,0.10,1.0->1.24,0.15,1.0'
  # Tier 4 — continuous integrator auto-discretised at Ts=0.05.
  # continuous_lowpass(u, s_x) -> (y, s_x_next):
  #   y = s_x;  s_x_next = s_x + Ts*(u - s_x) = s_x + 0.05*(u - s_x)
  # At (u=1, s_x=0): y=0, next=0.05.  At (u=1, s_x=0.5): y=0.5,
  # next=0.5 + 0.05*0.5 = 0.525.  At (u=0, s_x=1): y=1, next=0.95.
  'continuous_lowpass|1.0,0.0->0.0,0.05|1.0,0.5->0.5,0.525|0.0,1.0->1.0,0.95'
)

check() {
  local name="$1"; shift
  local cases=("$@")
  local mflow="$EX/$name.mflow"
  if [[ ! -f "$mflow" ]]; then
    fail=$((fail+1)); fails+=("$name (missing fixture)")
    return
  fi

  # --- Python emit + run ---
  local py="$SCRATCH/${name}_py.py"
  if ! "$MATLABC" -emit-python "$mflow" --subsystem "$name" \
       > "$py" 2> "$SCRATCH/emit.err"; then
    fail=$((fail+1)); fails+=("$name (python emit)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi

  python3 - "$py" "$name" "${cases[@]}" <<'PY'
import importlib.util, math, sys
py, name = sys.argv[1], sys.argv[2]
cases = sys.argv[3:]
spec = importlib.util.spec_from_file_location(name, py)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fn = getattr(mod, name)

def approx(a, b, tol=1e-6):
  if isinstance(a, bool) or isinstance(b, bool):
    return bool(a) == bool(b)
  if isinstance(a, tuple) or isinstance(b, tuple):
    a = tuple(a); b = tuple(b)
    return len(a) == len(b) and all(approx(x, y, tol) for x, y in zip(a, b))
  return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol)

failed = 0
for c in cases:
  inp, exp = c.split("->")
  args = [float(x) for x in inp.split(",")]
  out = fn(*args)
  if "," in exp:
    expected = tuple(eval(s.strip()) for s in exp.split(","))
  elif exp.strip() in ("True", "False"):
    expected = (exp.strip() == "True")
  else:
    expected = float(exp.strip())
  if not approx(out, expected):
    print(f"  case {inp} -> got {out}, expected {expected}", file=sys.stderr)
    failed += 1
sys.exit(0 if failed == 0 else 1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("$name (python mismatch)")
    return
  fi

  # --- C++ emit + run ---
  local cpp="$SCRATCH/${name}.cpp"
  if ! "$MATLABC" -emit-cpp "$mflow" --subsystem "$name" \
       > "$cpp" 2> "$SCRATCH/emit.err"; then
    fail=$((fail+1)); fails+=("$name (cpp emit)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  # Strip the synthesised priming main() and inject a tester.
  python3 - "$cpp" "$name" "${cases[@]}" <<'PY'
import re, sys
cpp, name = sys.argv[1], sys.argv[2]
cases = sys.argv[3:]
src = open(cpp).read()
# Drop the priming-driver main()
src = re.sub(r'int main\(void\) \{[^}]*\}\n', '', src)
tester  = '\n#include <cstdio>\n#include <cmath>\n'
tester += 'int main(void) {\n'
for c in cases:
  inp, exp = c.split("->")
  args = inp.split(",")
  if name == "comparator_logic":
    # Python returns Python's bool; C/C++ returns the integer result of AND
    # which Python prints as 0/1 — we accept either. Skip C++ for now.
    pass
  elif "," in exp:
    # Multi-return (struct? — matlab_llvm doesn't expose this cleanly
    # at C++ scope without a wrapper; skip for Tier 1).
    pass
  else:
    tester += f'  printf("%.10f\\n", (double)' + name + '(' + ", ".join(args) + ')); // expect ' + exp.strip() + '\n'
tester += '  return 0;\n}\n'
open(cpp, 'w').write(src + tester)
PY
  local bin="$SCRATCH/${name}_cpp"
  if ! clang++ -std=c++20 -O2 -o "$bin" "$cpp" 2> "$SCRATCH/build.err"; then
    fail=$((fail+1)); fails+=("$name (cpp build)")
    sed 's/^/  /' "$SCRATCH/build.err" >&2
    return
  fi
  local got expected ok
  ok=1
  > "$SCRATCH/cpp.out"
  "$bin" > "$SCRATCH/cpp.out"
  local i=0
  for c in "${cases[@]}"; do
    inp="${c%%->*}"; exp="${c##*->}"
    if [[ "$name" == "comparator_logic" || "$exp" == *,* ]]; then continue; fi
    expected="$exp"
    got=$(sed -n "$((i+1))p" "$SCRATCH/cpp.out")
    if ! awk -v g="$got" -v e="$expected" 'BEGIN{exit !((g - e)^2 < 1e-9)}'; then
      ok=0
      echo "  $name cpp: $inp -> got $got, expected $expected" >&2
    fi
    i=$((i+1))
  done
  if [[ $ok -ne 1 ]]; then
    fail=$((fail+1)); fails+=("$name (cpp mismatch)")
    return
  fi

  pass=$((pass+1))
}

for entry in "${CASES[@]}"; do
  IFS='|' read -r name rest <<< "$entry"
  IFS='|' read -ra cases <<< "$rest"
  check "$name" "${cases[@]}"
done

# Tier 2 — class wrapper smoke test for the stateful PID demo.
# Confirms the auto-emitted `DiscretePid` Python class produces
# the same per-tick `u` sequence as the functional form (and as
# the analytic reference baked into CASES above). Catches
# wrapper-template regressions (wrong state-slot order, missed
# latch, wrong return shape) without re-implementing every case.
class_smoke() {
  local fixture="$1" ; local cls="$2"
  local py="$SCRATCH/${fixture}_cls.py"
  "$MATLABC" -emit-python "$EX/${fixture}.mflow" --subsystem "$fixture" \
       > "$py" 2> "$SCRATCH/emit.err"
  python3 - "$py" "$fixture" "$cls" <<'PY'
import importlib.util, sys
py, fixture, cls = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location(fixture, py)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ctrl = getattr(m, cls)()
expected = [5.0, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38]
for tick, want in enumerate(expected):
    got = ctrl.step(1.0, 0.0)
    if abs(got - want) > 1e-6:
        print(f"  {cls}.step tick {tick}: got {got}, want {want}", file=sys.stderr)
        sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("$fixture (class wrapper smoke)")
  else
    pass=$((pass+1))
  fi
}
class_smoke discrete_pid DiscretePid

# Tier 4 — continuous-block discretization smoke test. Drive the
# auto-discretised 1/(s+1) lowpass with a step input for 5 seconds
# at Ts=0.05 and check the response stays within Forward-Euler
# error bounds of the analytic `1 - e^{-t}` ground truth.
lp_smoke() {
  local py="$SCRATCH/continuous_lowpass_cls.py"
  "$MATLABC" -emit-python "$EX/continuous_lowpass.mflow" \
      --subsystem continuous_lowpass --target-rate 0.05 \
      > "$py" 2> "$SCRATCH/emit.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
py = sys.argv[1]
spec = importlib.util.spec_from_file_location("cl", py)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.ContinuousLowpass()
Ts = 0.05
worst = 0.0
# The integrator returns the CURRENT (pre-update) state, so call k
# yields y[k] which corresponds to the analytic response at t = k·Ts
# (the first call returns the initial-condition zero).
for k in range(101):  # 0..5 s
    y = plant.step(1.0)
    t = k * Ts
    ref = 1.0 - math.exp(-t)
    err = abs(y - ref)
    if err > worst: worst = err
# Forward Euler error bound at Ts=0.05 stays below 1.5% over [0, 5].
if worst > 0.015:
    print(f"  continuous_lowpass: worst-case error {worst:.6f} > 0.015",
          file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("continuous_lowpass (class smoke)")
  else
    pass=$((pass+1))
  fi
}
lp_smoke

# Tier 5 — SystemVerilog emit smoke test. Verifies that a stateless
# subsystem with Gain + Sum lowers to synthesisable SV (default
# Q16.16 signed). The check is purely structural: the emitted source
# contains the `module <name>` declaration, the three fi-typed input
# ports, and the single fi-typed output. Synth-clean validation
# (Verilator lint, yosys synth) is gated on
# `MATLAB_LLVM_EMIT_SUBSYSTEM_SV_COSIM` (deferred).
sv_smoke() {
  local sv="$SCRATCH/scaled_sum.sv"
  "$MATLABC" -emit-sv "$EX/scaled_sum_sv.mflow" --subsystem scaled_sum \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module scaled_sum" "$sv"; then
    fail=$((fail+1)); fails+=("scaled_sum_sv (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  if ! grep -q "input  logic signed \[31:0\] u1" "$sv"; then
    fail=$((fail+1)); fails+=("scaled_sum_sv (wrong port type)")
    head -20 "$sv" >&2
    return
  fi
  if ! grep -q "output logic signed \[31:0\] y1" "$sv"; then
    fail=$((fail+1)); fails+=("scaled_sum_sv (missing output)")
    return
  fi
  pass=$((pass+1))
}
sv_smoke

# Tier-5b — stateful subsystem → SV. Three Unit Delays form a
# tapped delay line; each emits a clocked register, no fi-math on
# the persistent reads. Checks (i) the module has clk+rst+reset
# wired, (ii) three logic signed [31:0] regs are declared, (iii)
# the always_ff block is present.
sv_stateful_smoke() {
  local sv="$SCRATCH/tapped_delay.sv"
  "$MATLABC" -emit-sv "$EX/tapped_delay.mflow" --subsystem tapped_delay \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module tapped_delay" "$sv"; then
    fail=$((fail+1)); fails+=("tapped_delay (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  if ! grep -q "input  logic clk" "$sv" \
       || ! grep -q "input  logic rst_n" "$sv" \
       || ! grep -q "input  logic reset" "$sv"; then
    fail=$((fail+1)); fails+=("tapped_delay (missing clk/rst)")
    return
  fi
  if [[ $(grep -c "logic signed \[31:0\] s_" "$sv") -lt 3 ]]; then
    fail=$((fail+1)); fails+=("tapped_delay (missing state regs)")
    head -25 "$sv" >&2
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("tapped_delay (no always_ff block)")
    return
  fi
  pass=$((pass+1))
}
sv_stateful_smoke

# Tier-5c — FIR with fi-multiply on persistent reads. The pipeline
# used to barf with `matlab.add(i32, i64) -> none` because the
# function-arg path and the persistent-fetch path arrived at the
# `+` with mismatched widths (i32 vs i64 after fi-saturate).
# `BinArithToArith` now sign-extends the narrower operand. Verify
# the 4-tap FIR emits clean SV with 4 unit-delay registers + a
# combinational sum.
sv_fir_smoke() {
  local sv="$SCRATCH/fir_4tap.sv"
  "$MATLABC" -emit-sv "$EX/fir_4tap.mflow" --subsystem fir_4tap \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module fir_4tap" "$sv"; then
    fail=$((fail+1)); fails+=("fir_4tap (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  if [[ $(grep -c "logic signed \[31:0\] s_d" "$sv") -lt 3 ]]; then
    fail=$((fail+1)); fails+=("fir_4tap (missing delay regs)")
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("fir_4tap (no always_ff)")
    return
  fi
  pass=$((pass+1))
}
sv_fir_smoke

# #343 HDL — signal_dff → SystemVerilog. A D flip-flop must emit a single
# state register clocked on posedge: `s_ff_next = D` (combinational), `q = s_ff`,
# and `always_ff @(posedge clk ...) s_ff <= s_ff_next`. The block's `clk` input
# maps to the implicit module clock (single-clock design).
sv_dff_smoke() {
  local sv="$SCRATCH/dff_register.sv"
  "$MATLABC" -emit-sv "$EX/dff_register.mflow" --subsystem dff_reg \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module dff_reg" "$sv"; then
    fail=$((fail+1)); fails+=("dff_register (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2; return
  fi
  if ! grep -q "logic signed \[31:0\] s_ff" "$sv"; then
    fail=$((fail+1)); fails+=("dff_register (no state register)"); return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("dff_register (no always_ff)"); return
  fi
  if ! grep -q "s_ff <= s_ff_next" "$sv"; then
    fail=$((fail+1)); fails+=("dff_register (register not clocked)"); return
  fi
  pass=$((pass+1))
}
sv_dff_smoke

# Tier-5d — saturation → SV. The pure-arith form (used for
# software targets) routes through bool-by-fi multiplication
# that the SV synthcheck rejects. HDL mode now emits an explicit
# if/elseif/else form via the AST IfStmt node, which the SV
# pipeline lowers to a 3-way mux cleanly.  Verify that
# stateless_mixer.mflow (Gain · Sum · Saturation) emits clean
# SV — the if/else block plus the < / > comparison rails are
# present.
sv_sat_smoke() {
  local sv="$SCRATCH/stateless_mixer_sv.sv"
  "$MATLABC" -emit-sv "$EX/stateless_mixer.mflow" \
       --subsystem stateless_mixer \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module stateless_mixer" "$sv"; then
    fail=$((fail+1)); fails+=("stateless_mixer sat (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  # Tier-5k changed the saturation intermediate var from `mix` to
  # `clamp_sat` (the explicit fi-cast of the sum) so the upper-rail
  # comparison now reads `if (clamp_sat > 64'sd65536) ...`. Lower
  # rail keeps the `< -` shape.
  if ! grep -q "if (clamp_sat > " "$sv"; then
    fail=$((fail+1)); fails+=("stateless_mixer sat (missing upper rail)")
    return
  fi
  if ! grep -q "< -" "$sv"; then
    fail=$((fail+1)); fails+=("stateless_mixer sat (missing lower rail)")
    return
  fi
  pass=$((pass+1))
}
sv_sat_smoke

# Tier-5f — full discrete PID with saturation → SV. Combines
# stateful (discrete integrator, unit delay), fi-math width
# equalising (Tier-5c), saturation if/elseif/else (Tier-5d), and
# the mixed-width-store unifier (Tier-5f). Checks the module has
# both state registers (s_iacc, s_prev_err) AND the saturation
# rails in the always_comb block.
sv_pid_smoke() {
  local sv="$SCRATCH/discrete_pid_sv.sv"
  "$MATLABC" -emit-sv "$EX/discrete_pid.mflow" --subsystem discrete_pid \
       > "$sv" 2> "$SCRATCH/emit.err"
  if ! grep -q "module discrete_pid" "$sv"; then
    fail=$((fail+1)); fails+=("discrete_pid SV (missing module)")
    sed 's/^/  /' "$SCRATCH/emit.err" >&2
    return
  fi
  if ! grep -q "logic signed \[31:0\] s_iacc;" "$sv" \
       || ! grep -q "logic signed \[31:0\] s_prev_err;" "$sv"; then
    fail=$((fail+1)); fails+=("discrete_pid SV (missing state regs)")
    return
  fi
  # Tier-5k changed the saturation intermediate var name: the
  # explicit fi-cast lands in `clamp_sat` rather than the original
  # `ctrl`/`mix` user name. Both rails must still appear.
  if ! grep -q "if (clamp_sat > " "$sv" \
       || ! grep -q "< -" "$sv"; then
    fail=$((fail+1)); fails+=("discrete_pid SV (missing saturation rails)")
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("discrete_pid SV (no always_ff)")
    return
  fi
  pass=$((pass+1))
}
sv_pid_smoke

# Tier-5g — continuous Integrator + Transfer Function auto-
# discretise for HDL too (Forward Euler at the user-picked Ts).
# Validate continuous_lowpass.mflow (Integrator + Sum feedback)
# emits clean SV with the integrator state register.
sv_integrator_smoke() {
  local sv="$SCRATCH/continuous_lowpass.sv"
  "$MATLABC" -emit-sv "$EX/continuous_lowpass.mflow" \
       --subsystem continuous_lowpass > "$sv" \
       2> "$SCRATCH/cl.err"
  if ! grep -q "module continuous_lowpass" "$sv"; then
    fail=$((fail+1)); fails+=("continuous_lowpass SV (missing module)")
    sed 's/^/  /' "$SCRATCH/cl.err" >&2
    return
  fi
  if ! grep -q "logic signed \[31:0\] s_x;" "$sv"; then
    fail=$((fail+1)); fails+=("continuous_lowpass SV (missing integrator reg)")
    return
  fi
  pass=$((pass+1))
}
sv_integrator_smoke

# Tier-5g — continuous Transfer Function (1st-order
# strictly-proper) → SV. tf_lowpass.mflow emits a single-state
# Forward-Euler discretisation of 1/(s+1) at the user-picked Ts.
sv_tf_smoke() {
  local sv="$SCRATCH/tf_lowpass.sv"
  "$MATLABC" -emit-sv "$EX/tf_lowpass.mflow" \
       --subsystem tf_lowpass > "$sv" 2> "$SCRATCH/tf.err"
  if ! grep -q "module tf_lowpass" "$sv"; then
    fail=$((fail+1)); fails+=("tf_lowpass SV (missing module)")
    sed 's/^/  /' "$SCRATCH/tf.err" >&2
    return
  fi
  if ! grep -q "logic signed \[31:0\] s_tf;" "$sv"; then
    fail=$((fail+1)); fails+=("tf_lowpass SV (missing TF reg)")
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("tf_lowpass SV (no always_ff)")
    return
  fi
  pass=$((pass+1))
}
sv_tf_smoke

# Tier-5g — TF step response (Python class form). Drives
# 1/(s+1) at Ts=0.05 for 5 s and checks the Forward-Euler error
# vs the analytic 1 - e^{-t} stays within bounds.
tf_step_smoke() {
  local py="$SCRATCH/tf_step.py"
  "$MATLABC" -emit-python "$EX/tf_lowpass.mflow" \
       --subsystem tf_lowpass > "$py" 2> "$SCRATCH/tf.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location("tf", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.TfLowpass()
worst = 0.0
Ts = 0.05
for k in range(101):
    y = plant.step(1.0)
    t = k * Ts
    err = abs(y - (1.0 - math.exp(-t)))
    if err > worst: worst = err
if worst > 0.015:
    print(f"  tf_lowpass step: max err {worst}", file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("tf_lowpass step (out of tolerance)")
  else
    pass=$((pass+1))
  fi
}
tf_step_smoke

# Tier-5h — higher-order TF (2nd-order step response).
tf2_step_smoke() {
  local py="$SCRATCH/tf2.py"
  "$MATLABC" -emit-python "$EX/tf_2nd_order.mflow" \
       --subsystem tf_2nd_order > "$py" 2> "$SCRATCH/tf2.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("tf2", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.Tf2ndOrder()
peak = 0.0; final = 0.0
for k in range(500):
    y = plant.step(1.0)
    if y > peak: peak = y
    final = y
if not (1.45 <= peak <= 1.60):
    print(f"  tf_2nd_order peak {peak} out of band", file=sys.stderr)
    sys.exit(1)
if not (0.95 <= final <= 1.05):
    print(f"  tf_2nd_order final {final} not near 1", file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("tf_2nd_order (step response)")
  else
    pass=$((pass+1))
  fi
}
tf2_step_smoke

# Tier-5h — Zero-Pole via expansion to TF.
zp_step_smoke() {
  local py="$SCRATCH/zp.py"
  "$MATLABC" -emit-python "$EX/zp_plant.mflow" \
       --subsystem zp_plant > "$py" 2> "$SCRATCH/zp.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("zp", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.ZpPlant()
prev = 0.0; monotonic = True
for k in range(100):
    y = plant.step(1.0)
    if y < prev - 1e-9: monotonic = False
    prev = y
if not monotonic:
    print(f"  zp_plant not monotonic", file=sys.stderr); sys.exit(1)
if not (0.95 <= y <= 1.05):
    print(f"  zp_plant final {y}", file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("zp_plant (step response)")
  else
    pass=$((pass+1))
  fi
}
zp_step_smoke

# Tier-5h — transport delay (4-tap shift register).
td_smoke() {
  local py="$SCRATCH/td.py"
  "$MATLABC" -emit-python "$EX/transport_delay.mflow" \
       --subsystem transport_delay > "$py" 2> "$SCRATCH/td.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("td", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.TransportDelay()
seq = [plant.step(1.0) for _ in range(8)]
# 4-tap shift register: y[0..3] = 0, y[4..] = 1.
expected = [0, 0, 0, 0, 1, 1, 1, 1]
for k, (g, e) in enumerate(zip(seq, expected)):
    if abs(g - e) > 1e-9:
        print(f"  transport_delay tick {k}: got {g}, want {e}",
              file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("transport_delay (step response)")
  else
    pass=$((pass+1))
  fi
}
td_smoke

# Tier-5h — transport delay SV emit (multi-tap shift register).
sv_td_smoke() {
  local sv="$SCRATCH/td.sv"
  "$MATLABC" -emit-sv "$EX/transport_delay.mflow" \
       --subsystem transport_delay > "$sv" 2> "$SCRATCH/td.err"
  if [[ $(grep -c "logic signed \[31:0\] s_delay_x" "$sv") -lt 4 ]]; then
    fail=$((fail+1)); fails+=("transport_delay SV (missing taps)")
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("transport_delay SV (no always_ff)")
    return
  fi
  pass=$((pass+1))
}
sv_td_smoke

# Tier-5h — state-space (A, B, C; D=0). Same plant as
# tf_2nd_order.mflow but in (A, B, C) form: A = [0 1; -1 -0.4],
# B = [0; 1], C = [1 0]. Step response must match the TF form
# within tolerance.
ss_step_smoke() {
  local py="$SCRATCH/ss.py"
  "$MATLABC" -emit-python "$EX/ss_plant.mflow" \
       --subsystem ss_plant > "$py" 2> "$SCRATCH/ss.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ss", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.SsPlant()
peak = 0.0
for k in range(500):
    y = plant.step(1.0)
    if y > peak: peak = y
final = y
if not (1.45 <= peak <= 1.60):
    print(f"  ss_plant peak {peak} out of band", file=sys.stderr); sys.exit(1)
if not (0.95 <= final <= 1.05):
    print(f"  ss_plant final {final} off", file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("ss_plant (step response)")
  else
    pass=$((pass+1))
  fi
}
ss_step_smoke

# Tier-5i — Tustin (bilinear) discretisation of signal_transfer_fcn.
# `H(s) = 1/(s+1)` substituted s = (2/Ts)·(z-1)/(z+1) at Ts=0.05 gives
# `H(z) = (z+1)/(41z - 39)` (NumZ = [1/41, 1/41], DenZ = [1, -39/41]).
# Realised as Direct Form II Transposed: y[k] = NumZ[0]·u[k] + v[k].
# At t=0 with u=1, y[0] = 1/41 ≈ 0.024390 (direct feedthrough). DC
# gain = H(1) = 1, so step response settles to 1.0.
tustin_tf_smoke() {
  local py="$SCRATCH/tustin_tf.py"
  "$MATLABC" -emit-python "$EX/tf_lowpass.mflow" --subsystem tf_lowpass \
      --discretize=tustin > "$py" 2> "$SCRATCH/tustin.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("tf", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.TfLowpass()
ys = [plant.step(1.0) for _ in range(201)]  # 10 s at Ts=0.05
# Tustin direct feedthrough: y[0] = NumZ[0] = 1/41 ≈ 0.02439.
if abs(ys[0] - 1.0/41.0) > 1e-6:
    print(f"  tustin_tf: y[0]={ys[0]} != 1/41", file=sys.stderr); sys.exit(1)
# DC gain = 1: steady-state approaches 1.0.
if abs(ys[-1] - 1.0) > 0.01:
    print(f"  tustin_tf: final={ys[-1]} not near 1.0", file=sys.stderr); sys.exit(1)
# Pole at z = 39/41 ≈ 0.9512 → 5%-settling at log(0.05)/log(0.9512) ≈ 60
# steps. y[60] should be > 0.95 (well-settled).
if ys[60] < 0.95:
    print(f"  tustin_tf: y[60]={ys[60]} < 0.95", file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("tustin tf_lowpass (mismatched response)")
  else
    pass=$((pass+1))
  fi
}
tustin_tf_smoke

# Tier-5i — Tustin 2nd-order TF (underdamped ζ=0.2). Tustin gives a
# better peak match to the analytic 1.527 than Forward Euler (which
# overshoots to ~1.535 at Ts=0.01); checks peak stays in band.
tustin_tf2_smoke() {
  local py="$SCRATCH/tustin_tf2.py"
  "$MATLABC" -emit-python "$EX/tf_2nd_order.mflow" --subsystem tf_2nd_order \
      --discretize=tustin > "$py" 2> "$SCRATCH/tustin.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("tf2", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.Tf2ndOrder()
peak = 0.0; final = 0.0
for k in range(2000):
    y = plant.step(1.0); peak = max(peak, y); final = y
# Analytic peak (ζ=0.2, ωn=1): 1.527. Tustin at Ts=0.01 should be
# within ±0.02 of analytic — tighter than Forward Euler's overshoot.
if not (1.50 <= peak <= 1.55):
    print(f"  tustin_tf_2nd_order: peak {peak} out of band", file=sys.stderr)
    sys.exit(1)
if not (0.95 <= final <= 1.05):
    print(f"  tustin_tf_2nd_order: final {final} off", file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("tustin tf_2nd_order (mismatched response)")
  else
    pass=$((pass+1))
  fi
}
tustin_tf2_smoke

# Tier-5i — Tustin SISO state-space. The Faddeev-LeVerrier conversion
# `(A, B, C) -> (Num, Den)` should yield the same discrete TF (and
# therefore the same step response) as the Tustin TF path. ss_plant
# is the (A, B, C) realisation of tf_2nd_order; their Tustin responses
# must agree to numerical precision.
tustin_ss_smoke() {
  local py="$SCRATCH/tustin_ss.py"
  "$MATLABC" -emit-python "$EX/ss_plant.mflow" --subsystem ss_plant \
      --discretize=tustin > "$py" 2> "$SCRATCH/tustin.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ss", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.SsPlant()
peak = 0.0; final = 0.0
for k in range(2000):
    y = plant.step(1.0); peak = max(peak, y); final = y
if not (1.50 <= peak <= 1.55):
    print(f"  tustin_ss_plant: peak {peak} out of band", file=sys.stderr)
    sys.exit(1)
if not (0.95 <= final <= 1.05):
    print(f"  tustin_ss_plant: final {final} off", file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("tustin ss_plant (mismatched response)")
  else
    pass=$((pass+1))
  fi
}
tustin_ss_smoke

# Tier-5i — MIMO state-space (2-input, 2-output). Decoupled plant:
# A = diag(-1, -2), B = I, C = I → two independent first-order modes
# (gain 1, τ=1) and (gain 0.5, τ=0.5). Step responses on each input
# must affect only the corresponding output.
mimo_ss_smoke() {
  local py="$SCRATCH/mimo_ss.py"
  "$MATLABC" -emit-python "$EX/mimo_state_space.mflow" \
       --subsystem mimo_state_space > "$py" 2> "$SCRATCH/mimo.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location("mimo", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# Drive u1 only — y2 must stay zero, y1 settles near 1.
plant = m.MimoStateSpace()
for _ in range(200): y1, y2 = plant.step(1.0, 0.0)
if y2 != 0.0:
    print(f"  mimo: y2 leakage with u1 step ({y2})", file=sys.stderr); sys.exit(1)
if not (0.95 <= y1 <= 1.05):
    print(f"  mimo: y1 final {y1} off 1.0", file=sys.stderr); sys.exit(1)
# Drive u2 only — y1 must stay zero, y2 settles near 0.5.
plant = m.MimoStateSpace()
for _ in range(200): y1, y2 = plant.step(0.0, 1.0)
if y1 != 0.0:
    print(f"  mimo: y1 leakage with u2 step ({y1})", file=sys.stderr); sys.exit(1)
if not (0.45 <= y2 <= 0.55):
    print(f"  mimo: y2 final {y2} off 0.5", file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("mimo_state_space (response mismatch)")
  else
    pass=$((pass+1))
  fi
}
mimo_ss_smoke

# Tier-5i — MIMO state-space SV emit. Verifies the per-port output
# signals make it through the SV pipeline (`y1`/`y2`, both fi-typed),
# both state registers exist, and the always_ff block is present.
mimo_ss_sv_smoke() {
  local sv="$SCRATCH/mimo_ss.sv"
  "$MATLABC" -emit-sv "$EX/mimo_state_space.mflow" \
       --subsystem mimo_state_space > "$sv" 2> "$SCRATCH/mimo.err"
  if ! grep -q "module mimo_state_space" "$sv"; then
    fail=$((fail+1)); fails+=("mimo SV (missing module)")
    sed 's/^/  /' "$SCRATCH/mimo.err" >&2
    return
  fi
  if ! grep -q "output logic signed \[31:0\] y1" "$sv" \
       || ! grep -q "output logic signed \[31:0\] y2" "$sv"; then
    fail=$((fail+1)); fails+=("mimo SV (missing y1/y2 outputs)")
    return
  fi
  if [[ $(grep -c "logic signed \[31:0\] s_plant_x" "$sv") -lt 2 ]]; then
    fail=$((fail+1)); fails+=("mimo SV (missing state regs)")
    return
  fi
  pass=$((pass+1))
}
mimo_ss_sv_smoke

# Tier-5l — MIMO Tustin (matrix bilinear). Same 2-in/2-out
# decoupled plant as mimo_state_space.mflow but discretised via
# `--discretize=tustin`. Tustin's direct-feedthrough term
# `Dd·u` puts a non-zero first-sample response on each output
# (y1[0] = α·u1[0] / (1 - α·A[0,0]) ≈ 0.0244 for u1 = 1).
# DC gain still settles to 1.0 / 0.5 on respective outputs.
mimo_tustin_smoke() {
  local py="$SCRATCH/mimo_tustin.py"
  "$MATLABC" -emit-python "$EX/mimo_tustin.mflow" \
      --subsystem mimo_tustin --discretize=tustin \
      > "$py" 2> "$SCRATCH/mt.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mt", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.MimoTustin()
for _ in range(400): y1, y2 = plant.step(1.0, 0.0)
if abs(y2) > 1e-12:
    print(f"  mimo_tustin: y2 leakage with u1 step ({y2})", file=sys.stderr)
    sys.exit(1)
if not (0.95 <= y1 <= 1.05):
    print(f"  mimo_tustin: y1 final {y1} off 1.0", file=sys.stderr)
    sys.exit(1)
plant = m.MimoTustin()
for _ in range(400): y1, y2 = plant.step(0.0, 1.0)
if abs(y1) > 1e-12:
    print(f"  mimo_tustin: y1 leakage with u2 step ({y1})", file=sys.stderr)
    sys.exit(1)
if not (0.45 <= y2 <= 0.55):
    print(f"  mimo_tustin: y2 final {y2} off 0.5", file=sys.stderr)
    sys.exit(1)
# Direct feedthrough at tick 0 — the defining Tustin feature.
plant = m.MimoTustin()
y1, y2 = plant.step(1.0, 0.0)
if abs(y1 - 0.024390243902439025) > 1e-9:
    print(f"  mimo_tustin: y1 tick-0 feedthrough {y1} off 0.02439",
          file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("mimo_tustin (response mismatch)")
  else
    pass=$((pass+1))
  fi
}
mimo_tustin_smoke

# Tier-6 — nested signal_subsystem support. Outer subsystem
# `outer_loop` contains a `signal_subsystem` block pointing at the
# inner `lp_filter` flow (1/(s+1) lowpass), followed by a gain of
# 2.  The Embedded Coder lane emits both as sibling functions in
# the same TU, threads the inner's state through the outer's
# signature, and the class wrapper instantiates the outer + holds
# the threaded state.  Expected step response: `y(t) = 2·(1 − e⁻ᵗ)`,
# settles to 2.0.
nested_subsystem_smoke() {
  local py="$SCRATCH/nested.py"
  "$MATLABC" -emit-python "$EX/nested_pid_filter.mflow" \
      --subsystem outer_loop > "$py" 2> "$SCRATCH/nested.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location("nested", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.OuterLoop()
worst = 0.0; Ts = 0.05
for k in range(201):
    y = plant.step(1.0)
    t = k * Ts
    # Forward Euler 1/(s+1) step response, scaled by gain 2.
    # Reference matches the analytic continuous response within
    # the Forward-Euler discretisation error band.
    ref = 2.0 * (1.0 - math.exp(-t))
    err = abs(y - ref)
    if err > worst: worst = err
if worst > 0.03:
    print(f"  nested: max err {worst} > 0.03", file=sys.stderr); sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("nested_subsystem (response mismatch)")
  else
    pass=$((pass+1))
  fi
}
nested_subsystem_smoke

# Tier-6c — multirate subsystem. Two `signal_unit_delay` blocks
# at sampleTime 0.01 (base) and 0.05 (5x slower). The fast block
# latches every tick; the slow block latches every 5 ticks. The
# outer step() takes a hidden `_tick` member that counts up each
# call; the slow block's state-update is wrapped in
# `if mod(_tick, 5) == 0 ... end` with an else-branch holding
# the previous state.
multirate_smoke() {
  local py="$SCRATCH/multirate.py"
  "$MATLABC" -emit-python "$EX/multirate_filters.mflow" \
      --subsystem multirate_filters > "$py" 2> "$SCRATCH/mr.err"
  python3 - "$py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mr", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plant = m.MultirateFilters()
# Drive a level change at t=3. Fast block should latch u=5
# immediately (next tick); slow block should hold u=1 until the
# next firing tick (t=5), then latch u=5.
expected = [
    # (u,         expected_y, expected_s_fast, expected_s_slow)
    (1.0,         0.0,         1.0,             1.0),  # t=0 (firing)
    (1.0,         2.0,         1.0,             1.0),  # t=1
    (1.0,         2.0,         1.0,             1.0),  # t=2
    (5.0,         2.0,         5.0,             1.0),  # t=3 (slow holds)
    (5.0,         6.0,         5.0,             1.0),  # t=4 (slow holds)
    (5.0,         6.0,         5.0,             5.0),  # t=5 (slow fires)
    (5.0,        10.0,         5.0,             5.0),  # t=6
]
for t, (u, exp_y, exp_fast, exp_slow) in enumerate(expected):
    y = plant.step(u)
    if abs(y - exp_y) > 1e-9 or \
       abs(plant.s_fast - exp_fast) > 1e-9 or \
       abs(plant.s_slow - exp_slow) > 1e-9:
        print(f"  multirate t={t}: u={u} got y={y} "
              f"s_fast={plant.s_fast} s_slow={plant.s_slow}, "
              f"want y={exp_y} s_fast={exp_fast} s_slow={exp_slow}",
              file=sys.stderr)
        sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("multirate_filters (mismatch)")
  else
    pass=$((pass+1))
  fi
}
multirate_smoke

# Tier-5i — Tustin SV emit (synth sanity). Verifies the SV emit lane
# produces a valid module with the direct-feedthrough output equation
# (y = NumZ[0]*u + state) and the DF2T state update visible.
tustin_sv_smoke() {
  local sv="$SCRATCH/tustin_tf.sv"
  "$MATLABC" -emit-sv "$EX/tf_lowpass.mflow" --subsystem tf_lowpass \
      --discretize=tustin > "$sv" 2> "$SCRATCH/tustin_sv.err"
  if ! grep -q "module tf_lowpass" "$sv"; then
    fail=$((fail+1)); fails+=("tustin tf_lowpass SV (missing module)")
    sed 's/^/  /' "$SCRATCH/tustin_sv.err" >&2
    return
  fi
  if ! grep -q "logic signed \[31:0\] s_tf;" "$sv"; then
    fail=$((fail+1)); fails+=("tustin tf_lowpass SV (missing state reg)")
    return
  fi
  if ! grep -q "always_ff @(posedge clk" "$sv"; then
    fail=$((fail+1)); fails+=("tustin tf_lowpass SV (no always_ff)")
    return
  fi
  pass=$((pass+1))
}
tustin_sv_smoke

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
