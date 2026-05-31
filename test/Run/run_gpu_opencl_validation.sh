#!/usr/bin/env bash
# GPU Coder OpenCL device-validation lane — issue #25.
#
# Runs the OpenCL backend on a real device via the installed ICD loader.
# Validated on NVIDIA (its OpenCL ICD), but vendor-agnostic — works on
# AMD / Intel ICDs too.  HW-gated: SKIPs cleanly (exit 0) when no OpenCL
# ICD loader is present.
#
# Validates (fp64, ±1e-9):
#   1. The OpenCL runtime backend (runtime/gpu/opencl/runtime_gpu_opencl.cpp):
#      fp64 GEMM + the matlab_gpu_gemm dispatcher, via gpu_opencl_smoke.cpp.
#   2. The -emit-opencl bundle: emits the AXPY `scale` fixture, builds it
#      SDK-free (driver hand-declares the OpenCL API), runs it, checks output.
#
# Requires matlabc built with -DMATLAB_LLVM_GPU_OPENCL=ON; (re)configures +
# builds if needed.  Usage: run_gpu_opencl_validation.sh [build-dir]
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-${BUILD_DIR:-$ROOT/build}}"

skip() { echo "SKIP (opencl validation): $*"; exit 0; }
fail() { echo "FAIL (opencl validation): $*"; exit 1; }

# ---- 1. HW gate: find an ICD loader -----------------------------------------
OCL_LIB=""
for cand in /usr/lib/x86_64-linux-gnu/libOpenCL.so* /lib/x86_64-linux-gnu/libOpenCL.so* \
            /usr/lib64/libOpenCL.so*; do
  [[ -e "$cand" ]] && { OCL_LIB="$cand"; break; }
done
[[ -n "$OCL_LIB" ]] || skip "no OpenCL ICD loader (libOpenCL) found"
if [[ ! -e /etc/OpenCL/vendors ]] || [[ -z "$(ls -A /etc/OpenCL/vendors 2>/dev/null)" ]]; then
  skip "no OpenCL ICD vendor registered in /etc/OpenCL/vendors"
fi
echo "opencl validation: ICD loader = $OCL_LIB"

# ---- 2. Ensure matlabc is built with the OpenCL backend ---------------------
cache="$BUILD/CMakeCache.txt"
if [[ ! -f "$cache" ]] || ! grep -q "^MATLAB_LLVM_GPU_OPENCL:BOOL=ON" "$cache"; then
  echo "opencl validation: configuring $BUILD with -DMATLAB_LLVM_GPU_OPENCL=ON"
  cmake -S "$ROOT" -B "$BUILD" -G Ninja -DMATLAB_LLVM_GPU_OPENCL=ON >/dev/null \
    || fail "cmake configure failed"
fi
cmake --build "$BUILD" --target matlabc >/dev/null 2>&1 || fail "matlabc build failed"
MATLABC="$BUILD/matlabc"
[[ -x "$MATLABC" ]] || fail "matlabc not found at $MATLABC"
RT_LIB="$BUILD/libMatlabRuntime.a"
[[ -f "$RT_LIB" ]] || fail "MatlabRuntime static lib not found ($RT_LIB)"
# Use the loader the build linked, if recorded.
cache_lib="$(grep '^MATLAB_OPENCL_LIB:' "$cache" 2>/dev/null | head -1 | cut -d= -f2-)"
[[ -n "$cache_lib" && -e "$cache_lib" ]] && OCL_LIB="$cache_lib"

TMP="$(mktemp -d -t gpu-opencl-val.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
pass=0; fail_n=0

# ---- 3. Runtime-backend smoke (GEMM + dispatcher) ---------------------------
echo "== runtime backend smoke =="
SMOKE_BIN="$TMP/opencl_smoke"
if g++ -O2 -std=c++17 -I"$ROOT/runtime" \
     "$ROOT/test/Run/gpu_opencl_smoke.cpp" \
     "$ROOT/runtime/gpu/opencl/runtime_gpu_opencl.cpp" \
     "$RT_LIB" "$OCL_LIB" -lm -ldl -lpthread \
     -o "$SMOKE_BIN" 2>"$TMP/smoke_build.err"; then
  if MATLAB_GPU_TARGET=opencl MATLAB_GPU_GEMM_MIN=1 "$SMOKE_BIN"; then
    pass=$((pass+1))
  else
    echo "FAIL: smoke reported a numeric mismatch"; fail_n=$((fail_n+1))
  fi
else
  echo "FAIL: smoke build failed"; sed 's/^/  /' "$TMP/smoke_build.err" | head; fail_n=$((fail_n+1))
fi

# ---- 4. -emit-opencl bundle end-to-end (AXPY scale fixture) -----------------
echo "== emit-opencl bundle (AXPY) =="
cat > "$TMP/scale.m" << 'EOF'
function y = scale(x, n)
    coder.gpu.kernelfun();
    y = zeros(1, n);
    for i = 1:n
        y(i) = x * i;
    end
end
EOF
( cd "$TMP" && "$MATLABC" -emit-opencl scale.m ) >/dev/null 2>"$TMP/emit.err" \
  || { echo "FAIL: -emit-opencl errored"; sed 's/^/  /' "$TMP/emit.err"; fail_n=$((fail_n+1)); }
BUNDLE="$TMP/scale_opencl"
if [[ -f "$BUNDLE/scale_kernel.cl" ]]; then
  if ! grep -q "out\[(int)(iv) - 1\] = (x \* iv);" "$BUNDLE/scale_kernel.cl"; then
    echo "FAIL: emitted kernel body did not translate the AXPY expression"
    fail_n=$((fail_n+1))
  elif make -C "$BUNDLE" OPENCL_LIB="$OCL_LIB" >/dev/null 2>"$TMP/make.err"; then
    # scale demo x=2.0, n=8 → out[i] = 2*(i+1) = 2 4 6 8 10 12 14 16
    got="$(cd "$BUNDLE" && ./scale_opencl 8 | tr '\n' ' ' | sed 's/ *$//')"
    want="2 4 6 8 10 12 14 16"
    if [[ "$got" == "$want" ]]; then
      echo "  bundle output OK: $got"; pass=$((pass+1))
    else
      echo "FAIL: bundle output mismatch"; echo "  got:  $got"; echo "  want: $want"
      fail_n=$((fail_n+1))
    fi
  else
    echo "FAIL: bundle build failed"; sed 's/^/  /' "$TMP/make.err" | head; fail_n=$((fail_n+1))
  fi
else
  echo "FAIL: bundle missing scale_kernel.cl"; fail_n=$((fail_n+1))
fi

echo "opencl validation: passed $pass, failed $fail_n"
[[ $fail_n -eq 0 ]] || exit 1
echo "PASS (opencl validation)"
