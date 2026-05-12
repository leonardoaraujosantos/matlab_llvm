#!/usr/bin/env bash
# Tier-10 cosim lane.  For each .m under examples/verilog_a/:
#   1. Compile via matlabc + clang.
#   2. Run the binary (writes the .va file).
#   3. Pipe the .va through scripts/va_cosim.sh, which delegates to
#      ngspice + OpenVAF (preferred) or Xyce + ADMS.
#
# Skips cleanly (exit 0) when no cosim toolchain is installed.
# Wired into CTest via MATLAB_LLVM_WITH_VA_COSIM.
#
# Usage: run_cosim.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Detect toolchain availability up-front so we can skip without
# rebuilding the examples.
if ! command -v ngspice >/dev/null 2>&1 && \
   ! command -v Xyce    >/dev/null 2>&1; then
  echo "skip: neither ngspice nor Xyce is on PATH." >&2
  exit 0
fi
if command -v ngspice >/dev/null 2>&1 && \
   ! command -v openvaf >/dev/null 2>&1; then
  echo "skip: ngspice present but OpenVAF missing." >&2
  echo "      Install OpenVAF (https://openvaf.semimod.de/) to" >&2
  echo "      enable .va cosim against ngspice." >&2
  exit 0
fi

CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
CXX="${CXX:-${CLANG}++}"
EXAMPLES_DIR="$ROOT/examples/verilog_a"
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
  "$ROOT/runtime/runtime_comm.cpp"
  "$ROOT/runtime/runtime_prop.cpp"
  "$ROOT/runtime/runtime_rf.cpp"
)

# Cosim is selective: only a subset of examples have 1-in/1-out
# port topology that fits the canonical AC-sweep netlist template.
# Composite blocks with extra ports (AM modulator, I/Q modulator,
# comparator, DAC, etc.) need per-block testbenches — out of scope
# for this generic lane.
SUITABLE=(
  rc_lowpass_tf
  biquad_butterworth
  resonant_bpf_zpk
  low_pass_filter
  rc_lowpass_ss
  biquad_ss_controllable
  butter3_observable
  rf_rational_writeva
)

WORKDIR="$(mktemp -d -t va_cosim.XXXXXX)"
trap "rm -rf '$WORKDIR'" EXIT

fail=0; ran=0
for base in "${SUITABLE[@]}"; do
  m="$EXAMPLES_DIR/${base}.m"
  [[ -e "$m" ]] || continue

  tmpll="$WORKDIR/${base}.ll"
  tmpbin="$WORKDIR/${base}.out"
  if ! "$MATLABC" -emit-llvm "$m" > "$tmpll" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-llvm" >&2
    fail=$((fail+1)); continue
  fi
  if ! "$CXX" -Wno-override-module "$tmpll" "${RUNTIME_SRCS[@]}" \
        -I"$ROOT/runtime" -o "$tmpbin" 2>/dev/null; then
    echo "FAIL $base: clang link" >&2
    fail=$((fail+1)); continue
  fi
  (cd "$WORKDIR" && "$tmpbin" >/dev/null 2>&1) || {
    echo "FAIL $base: binary exit nonzero" >&2
    fail=$((fail+1)); continue
  }

  # Find the .va that the example wrote.
  va_path=""
  for cand in "$WORKDIR"/*.va; do
    [[ -e "$cand" ]] || continue
    if [[ "$(basename "${cand%.va}")" == "$base"* ]]; then
      va_path="$cand"; break
    fi
  done
  # Fallback: take the most-recently-modified .va.
  if [[ -z "$va_path" ]]; then
    va_path="$(ls -t "$WORKDIR"/*.va 2>/dev/null | head -1)"
  fi
  [[ -e "$va_path" ]] || { echo "FAIL $base: no .va emitted" >&2; fail=$((fail+1)); continue; }

  if "$ROOT/scripts/va_cosim.sh" "$va_path" 2>&1 | grep -q "^ok:"; then
    ran=$((ran+1))
  else
    echo "FAIL $base: cosim" >&2
    fail=$((fail+1))
  fi
done

echo "cosim ran=$ran fail=$fail"
exit $(( fail > 0 ? 1 : 0 ))
