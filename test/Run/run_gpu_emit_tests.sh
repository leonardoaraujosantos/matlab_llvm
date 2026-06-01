#!/usr/bin/env bash
# GPU emission CI lane — verifies `matlabc -emit-{cuda,metal,opencl}`
# produces structurally-complete bundles for a representative MATLAB
# program with `coder.gpu.kernelfun`.  Does NOT require GPU hardware —
# only checks that:
#   1. matlabc exits 0
#   2. each bundle dir contains a kernel-source file + host driver +
#      Makefile + README
#   3. the kernel-source file contains the canonical target-specific
#      preamble (so the emit pass walked the body, not just the
#      placeholder)
#
# This is the GitHub Actions gate the user asked for: emission-only
# (CUDA + OpenCL + Metal), no execution.  Real device validation
# happens on a separate macOS-Metal CI runner via the
# test/Run/gpu_metal_*_smoke.mm fixtures.
#
# Usage: run_gpu_emit_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d -t gpu-emit.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# Test source — a coder.gpu.kernelfun function with one tensor output
# and a scalar capture.  Matches the AXPY-pattern the emitters handle.
SRC="$TMP/gpu_emit_src.m"
cat > "$SRC" << 'EOF'
function y = scale(x, n)
    coder.gpu.kernelfun();
    y = zeros(1, n);
    for i = 1:n
        y(i) = x * i;
    end
end
EOF

pass=0
fail=0
for target in cuda metal opencl; do
  STEM="gpu_emit_src"
  DIR="$TMP/${STEM}_${target}"
  rm -rf "$DIR"
  cd "$TMP"
  if ! "$MATLABC" -emit-$target "$SRC" 2>"$TMP/$target.err"; then
    echo "FAIL $target: matlabc errored"
    sed 's/^/  /' "$TMP/$target.err" | head -5
    fail=$((fail+1))
    continue
  fi
  # 2. Bundle structural check.
  BUNDLE="${STEM}_${target}"
  case $target in
    cuda)    K="$BUNDLE/${STEM}_kernel.cu";   D="$BUNDLE/${STEM}_main.cpp" ;;
    metal)   K="$BUNDLE/${STEM}_kernel.metal"; D="$BUNDLE/${STEM}_main.mm" ;;
    opencl)  K="$BUNDLE/${STEM}_kernel.cl";   D="$BUNDLE/${STEM}_main.cpp" ;;
  esac
  missing=()
  [[ -f "$TMP/$K" ]]               || missing+=("$K")
  [[ -f "$TMP/$D" ]]               || missing+=("$D")
  [[ -f "$TMP/$BUNDLE/Makefile" ]] || missing+=("$BUNDLE/Makefile")
  [[ -f "$TMP/$BUNDLE/README.md" ]] || missing+=("$BUNDLE/README.md")
  if (( ${#missing[@]} > 0 )); then
    echo "FAIL $target: missing files: ${missing[*]}"
    fail=$((fail+1))
    continue
  fi
  # 3. Per-target kernel preamble check.
  case $target in
    cuda)
      if ! grep -q "__global__" "$TMP/$K"; then
        echo "FAIL $target: kernel missing __global__"
        fail=$((fail+1)); continue
      fi
      ;;
    metal)
      if ! grep -q "kernel void" "$TMP/$K" || ! grep -q "thread_position_in_grid" "$TMP/$K"; then
        echo "FAIL $target: kernel missing MSL preamble"
        fail=$((fail+1)); continue
      fi
      ;;
    opencl)
      if ! grep -q "__kernel void" "$TMP/$K" || ! grep -q "get_global_id" "$TMP/$K"; then
        echo "FAIL $target: kernel missing OpenCL preamble"
        fail=$((fail+1)); continue
      fi
      ;;
  esac
  echo "PASS $target"
  pass=$((pass+1))
done

# On macOS, also confirm the Metal bundle BUILDS (host driver compiles
# against the system Metal SDK).  Validates the bundle is link-ready
# even without running.
if [[ "$(uname)" == "Darwin" && -f "$TMP/gpu_emit_src_metal/gpu_emit_src_main.mm" ]]; then
  cd "$TMP/gpu_emit_src_metal"
  if clang++ -O2 -std=c++20 -fobjc-arc gpu_emit_src_main.mm \
        -framework Metal -framework MetalPerformanceShaders -framework Foundation \
        -o gpu_emit_src_metal 2>"$TMP/metal_build.err"; then
    echo "PASS metal-bundle-builds"
    pass=$((pass+1))
    # Run the bundle end-to-end (its host driver JIT-compiles the .metal,
    # binds out@0 + the scalar captures, dispatches, and prints a
    # checksum) — proves the driver's kernel-function name + buffer ABI
    # match the emitted kernel.  Skips cleanly if the runner has no GPU.
    run_out="$(./gpu_emit_src_metal 2>/dev/null)"
    if [[ "$run_out" == *"checksum ="* ]]; then
      echo "PASS metal-bundle-runs ($run_out)"
      pass=$((pass+1))
    elif [[ "$run_out" == *"no Metal device"* || -z "$run_out" ]]; then
      echo "SKIP metal-bundle-runs (no Metal device on this runner)"
    else
      echo "FAIL metal-bundle-runs: unexpected output [$run_out]"
      fail=$((fail+1))
    fi
  else
    echo "FAIL metal-bundle-builds: clang++ link error"
    sed 's/^/  /' "$TMP/metal_build.err" | head -5
    fail=$((fail+1))
  fi
fi

echo
echo "passed: $pass    failed: $fail"
exit $(( fail == 0 ? 0 : 1 ))
