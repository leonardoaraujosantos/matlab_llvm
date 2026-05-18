#!/usr/bin/env bash
# Tier-7 — whole-diagram emit smoke tests.  Each fixture is a
# `.mflow` model whose entry flow is a self-contained diagram
# (sources + sinks + subsystems).  `matlabc -emit-{python,cpp}
# <model.mflow>` (without `--subsystem`) emits a top-level
# `simulate()` that runs the time loop and returns per-sink log
# arrays.  Each test runs the emit and verifies numerical
# behaviour against an analytic reference.
#
# Usage: run_diagram_tests.sh <path-to-matlabc>
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
export PYTHONPATH="$ROOT/runtime/shim${PYTHONPATH:+:$PYTHONPATH}"

pass=0; fail=0
fails=()

# Closed-loop step demo: source step at t=0.5 → err → 1/(s+1)
# plant → fed back into err.  DC closed-loop gain = 1/2, so the
# plant output settles to ~0.5 by t=4 s.
closed_loop_smoke() {
  local py="$SCRATCH/whole_pid.py"
  "$MATLABC" -emit-python "$EX/whole_pid_loop.mflow" > "$py" \
      2> "$SCRATCH/whole_pid.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
import numpy as np
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
log_ref, log_plant, log_err, t_log = m.simulate()
plant = np.array(log_plant).flatten()
# Forward-Euler closed loop with TF 1/(s+1), unity negative
# feedback ⇒ effective time constant 1/2 s, DC gain 0.5.  After
# 3.5 s (step at 0.5 s, sim ends 4 s) the response is essentially
# settled.
final = plant[-1]
if not (0.49 <= final <= 0.51):
    print(f"  whole_pid: plant final {final} not near 0.5", file=sys.stderr)
    sys.exit(1)
# Step delay: plant[0..9] should all be zero (step fires at idx 10).
if any(abs(v) > 1e-9 for v in plant[:10]):
    print(f"  whole_pid: plant nonzero before step time", file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("whole_pid_loop (response mismatch)")
  else
    pass=$((pass+1))
  fi
}
closed_loop_smoke

# Sine source → 1st-order LP demo.  Peak input amp=1 at t=0.25;
# LP at ω_c = 1 rad/s attenuates the 2π rad/s tone significantly,
# so the LP peak amplitude is much smaller and delayed.
sine_lp_smoke() {
  local py="$SCRATCH/whole_sine.py"
  "$MATLABC" -emit-python "$EX/whole_sine_lp.mflow" > "$py" \
      2> "$SCRATCH/whole_sine.err"
  python3 - "$py" <<'PY'
import importlib.util, math, sys
import numpy as np
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
log_in, log_out, t_log = m.simulate()
in_arr = np.array(log_in).flatten()
out_arr = np.array(log_out).flatten()
# Sine input — peak amplitude should be near 1.0.
peak_in = float(np.max(in_arr))
if not (0.99 <= peak_in <= 1.01):
    print(f"  whole_sine: sine peak {peak_in} off 1.0", file=sys.stderr)
    sys.exit(1)
# LP output — significantly attenuated; peak < 0.5 in steady state.
peak_out = float(np.max(np.abs(out_arr[100:])))
if peak_out > 0.5:
    print(f"  whole_sine: LP output peak {peak_out} should be < 0.5",
          file=sys.stderr)
    sys.exit(1)
# Sanity: LP output isn't flat-zero.
if peak_out < 0.05:
    print(f"  whole_sine: LP output peak {peak_out} too small",
          file=sys.stderr)
    sys.exit(1)
PY
  if [[ $? -ne 0 ]]; then
    fail=$((fail+1)); fails+=("whole_sine_lp (response mismatch)")
  else
    pass=$((pass+1))
  fi
}
sine_lp_smoke

# Tier-7e — per-language emit smoke. Each fixture is also emitted
# through -emit-c / -emit-cpp / -emit-typescript; we only check
# that emit succeeds (rc=0) and stdout is non-empty. Full
# compile+run lives in the per-subsystem cosim lane; this gate is
# a shorter "did the lowering pipeline stay green" sentinel.
per_language_smoke() {
  local fixture="$1"
  local mflow="$EX/${fixture}.mflow"
  for lang in c cpp ts; do
    case "$lang" in
      c)   flag="-emit-c"; out="$SCRATCH/${fixture}.c";;
      cpp) flag="-emit-cpp"; out="$SCRATCH/${fixture}.cpp";;
      ts)  flag="-emit-typescript"; out="$SCRATCH/${fixture}.ts";;
    esac
    "$MATLABC" "$flag" "$mflow" > "$out" \
        2> "$SCRATCH/${fixture}.${lang}.err"
    if [[ $? -ne 0 ]]; then
      fail=$((fail+1)); fails+=("$fixture ${flag} (emit failed)")
      sed 's/^/  /' "$SCRATCH/${fixture}.${lang}.err" >&2
      continue
    fi
    if [[ ! -s "$out" ]]; then
      fail=$((fail+1)); fails+=("$fixture ${flag} (empty output)")
      continue
    fi
    pass=$((pass+1))
  done
}
per_language_smoke whole_pid_loop
per_language_smoke whole_sine_lp

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
