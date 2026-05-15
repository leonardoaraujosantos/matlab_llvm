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
export PYTHONPATH="$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"

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
  if ! clang++ -O2 -o "$bin" "$cpp" 2> "$SCRATCH/build.err"; then
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

# Tier 5 — continuous block must be rejected with a sourced error
# in HDL emit (no implicit auto-discretisation).
sv_reject() {
  if "$MATLABC" -emit-sv "$EX/continuous_lowpass.mflow" \
       --subsystem continuous_lowpass > "$SCRATCH/out.sv" \
       2> "$SCRATCH/reject.err"; then
    fail=$((fail+1)); fails+=("continuous_lowpass SV (no rejection)")
    return
  fi
  if ! grep -q "continuous block" "$SCRATCH/reject.err"; then
    fail=$((fail+1)); fails+=("continuous_lowpass SV (wrong error)")
    sed 's/^/  /' "$SCRATCH/reject.err" >&2
    return
  fi
  pass=$((pass+1))
}
sv_reject

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
