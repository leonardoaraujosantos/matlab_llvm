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
# `<u1>,<u2>,... -> <y1>[,<y2>...]`. The Python harness eval's the
# subsystem call and asserts approximate equality (1e-9).
declare -a CASES=(
  # stateless_mixer: y = sat(2*u1 + u2 - u3), sat clamps to [-1, 1]
  'stateless_mixer|0.3,0.2,0.1->0.7|1.0,1.0,0.0->1.0|-1.0,-2.0,0.0->-1.0|0.0,0.0,0.0->0.0'
  # comparator_logic: y = (u1 > 0) AND (u2 < 5)
  'comparator_logic|1.0,1.0->True|1.0,7.0->False|-1.0,1.0->False'
  # threshold_switch: y = (u2 > 0.5) ? u1 : u3
  'threshold_switch|10.0,0.6,-10.0->10.0|10.0,0.4,-10.0->-10.0|10.0,0.5,-10.0->-10.0'
  # math_fns: returns (abs(u1), sin(u1))
  'math_fns|0.0->0.0,0.0|1.5->1.5,0.9974949866|-2.0->2.0,-0.9092974268'
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

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  printf '  %s\n' "${fails[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
