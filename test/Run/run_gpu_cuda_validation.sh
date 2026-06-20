#!/usr/bin/env bash
# GPU Coder CUDA device-validation lane — issue #25.
#
# Unlike run_gpu_emit_tests.sh (emission-only, no HW), this lane runs the
# CUDA backend on a *real* NVIDIA GPU.  It is HW-gated: when no NVIDIA
# device is present it SKIPs cleanly (exit 0) so it can sit in CI behind a
# self-hosted-runner label without breaking HW-less runs.
#
# What it validates on hardware (fp64, ±1e-9):
#   1. The CUDA runtime backend (runtime/gpu/cuda/runtime_gpu_cuda.cpp):
#      cuBLAS Dgemm + NVRTC AXPY + the matlab_gpu_gemm dispatcher, via
#      test/Run/gpu_cuda_smoke.cpp.
#   2. The -emit-cuda bundle: emits the AXPY `scale` fixture, builds it
#      nvcc-free (host driver JIT-compiles the kernel via NVRTC), runs it,
#      and checks the numeric output.
#
# Requires matlabc built with -DMATLAB_LLVM_GPU_CUDA=ON; the script will
# (re)configure + build that target if the build dir lacks the CUDA cache
# entries.  CUDA libs are discovered by CMake (system CUDA preferred, pip
# wheels otherwise) and read back from CMakeCache.txt here.
#
# Usage: run_gpu_cuda_validation.sh [build-dir]
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-${BUILD_DIR:-$ROOT/build}}"

skip() { echo "SKIP (cuda validation): $*"; exit 0; }
fail() { echo "FAIL (cuda validation): $*"; exit 1; }

# ---- 1. HW gate -----------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  skip "nvidia-smi not found — no NVIDIA driver"
fi
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU 0"; then
  skip "no NVIDIA GPU reported by nvidia-smi"
fi
echo "cuda validation: device = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---- 2. Ensure matlabc is built with the CUDA backend ---------------------
cache="$BUILD/CMakeCache.txt"
need_configure=0
if [[ ! -f "$cache" ]] || ! grep -q "^MATLAB_LLVM_GPU_CUDA:BOOL=ON" "$cache"; then
  need_configure=1
fi
if (( need_configure )); then
  echo "cuda validation: configuring $BUILD with -DMATLAB_LLVM_GPU_CUDA=ON"
  cmake -S "$ROOT" -B "$BUILD" -G Ninja -DMATLAB_LLVM_GPU_CUDA=ON >/dev/null \
    || fail "cmake configure failed"
fi
cmake --build "$BUILD" --target matlabc >/dev/null 2>&1 \
  || fail "matlabc build failed"
MATLABC="$BUILD/matlabc"
[[ -x "$MATLABC" ]] || fail "matlabc not found at $MATLABC"

# Read the CUDA toolchain CMake discovered.
cache_val() { grep "^$1:" "$cache" | head -1 | cut -d= -f2-; }
CUDA_INCS="$(cache_val MATLAB_CUDA_INCLUDE_DIRS)"
CUDA_LIBS_RAW="$(cache_val MATLAB_CUDA_LIBRARIES)"
CUDA_RPATH_RAW="$(cache_val MATLAB_CUDA_RPATH)"
[[ -n "$CUDA_LIBS_RAW" ]] || fail "no CUDA libraries in $cache"

INC_FLAGS=""; for d in ${CUDA_INCS//;/ }; do INC_FLAGS="$INC_FLAGS -I$d"; done
LIB_FLAGS=""; for l in ${CUDA_LIBS_RAW//;/ }; do LIB_FLAGS="$LIB_FLAGS $l"; done
RPATH_FLAGS=""; for r in ${CUDA_RPATH_RAW//;/ }; do RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath,$r"; done

RT_LIB="$BUILD/libMatlabRuntime.a"
[[ -f "$RT_LIB" ]] || fail "MatlabRuntime static lib not found ($RT_LIB)"

TMP="$(mktemp -d -t gpu-cuda-val.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
pass=0; fail_n=0

# ---- 3. Runtime-backend smoke (cuBLAS + NVRTC + dispatcher) ---------------
echo "== runtime backend smoke =="
SMOKE_BIN="$TMP/cuda_smoke"
# shellcheck disable=SC2086
if g++ -O2 -std=c++17 -I"$ROOT/runtime" $INC_FLAGS \
     "$ROOT/test/Run/gpu_cuda_smoke.cpp" \
     "$ROOT/runtime/gpu/cuda/runtime_gpu_cuda.cpp" \
     "$RT_LIB" $LIB_FLAGS -lm -ldl -lpthread $RPATH_FLAGS \
     -o "$SMOKE_BIN" 2>"$TMP/smoke_build.err"; then
  if MATLAB_GPU_TARGET=cuda MATLAB_GPU_GEMM_MIN=1 "$SMOKE_BIN"; then
    pass=$((pass+1))
  else
    echo "FAIL: smoke reported a numeric mismatch"; fail_n=$((fail_n+1))
  fi
else
  echo "FAIL: smoke build failed"; sed 's/^/  /' "$TMP/smoke_build.err" | head; fail_n=$((fail_n+1))
fi

# ---- 4. -emit-cuda bundle end-to-end (AXPY scale fixture) -----------------
echo "== emit-cuda bundle (AXPY) =="
cat > "$TMP/scale.m" << 'EOF'
function y = scale(x, n)
    coder.gpu.kernelfun();
    y = zeros(1, n);
    for i = 1:n
        y(i) = x * i;
    end
end
EOF
( cd "$TMP" && "$MATLABC" -emit-cuda scale.m ) >/dev/null 2>"$TMP/emit.err" \
  || { echo "FAIL: -emit-cuda errored"; sed 's/^/  /' "$TMP/emit.err"; fail_n=$((fail_n+1)); }
BUNDLE="$TMP/scale_cuda"
if [[ -f "$BUNDLE/scale_kernel.cu" ]]; then
  if ! grep -q "out\[(int)(iv) - 1\] = (x \* iv);" "$BUNDLE/scale_kernel.cu"; then
    echo "FAIL: emitted kernel body did not translate the AXPY expression"
    fail_n=$((fail_n+1))
  elif make -C "$BUNDLE" CUDA_INC="$INC_FLAGS" \
            CUDA_LIBS="$LIB_FLAGS $RPATH_FLAGS" >/dev/null 2>"$TMP/make.err"; then
    # The bundle driver runs the NVRTC-JIT'd kernel on the GPU and prints a
    # checksum of the result: for n=8 it computes y(i) = x*i with x=n=8, so
    # the checksum is sum_{i=1..8} 8*i = 8 * 36 = 288.0000.  (The driver was
    # changed from dumping the element list to a single checksum line; this
    # check follows that.)  Run from inside the bundle dir — the driver reads
    # scale_kernel.cu by relative path (matches the bundle README).
    got="$(cd "$BUNDLE" && ./scale_cuda 8 | tr '\n' ' ' | sed 's/ *$//')"
    want="scale: checksum = 288.0000"
    if [[ "$got" == "$want" ]]; then
      echo "  bundle output OK: $got"
      pass=$((pass+1))
    else
      echo "FAIL: bundle output mismatch"; echo "  got:  $got"; echo "  want: $want"
      fail_n=$((fail_n+1))
    fi
  else
    echo "FAIL: bundle build failed"; sed 's/^/  /' "$TMP/make.err" | head
    fail_n=$((fail_n+1))
  fi
else
  echo "FAIL: bundle missing scale_kernel.cu"; fail_n=$((fail_n+1))
fi

# ---- 5. In-process gpuArray dispatch (#335 Tier C) ------------------------
# A plain `Ag = gpuArray(A); Cg = Ag*Bg; gather(Cg)` program, run through the
# JIT/-repl lane with MATLAB_GPU_TARGET=auto, must (a) escalate to the CUDA
# device (MATLAB_GPU_TRACE shows the dispatch) and (b) be numerically correct.
echo "== in-process gpuArray dispatch (Tier C) =="
cat > "$TMP/gpurun.m" << 'EOF'
A = ones(256, 256);
B = ones(256, 256);
Cg = gpuArray(A) * gpuArray(B);
C = gather(Cg);
fprintf('gpuarray_gemm checksum = %.1f\n', C(1, 1));
EOF
GP_OUT="$(MATLAB_GPU_TARGET=auto MATLAB_GPU_TRACE=1 MATLAB_GPU_GEMM_MIN=1 \
          "$MATLABC" -repl < "$TMP/gpurun.m" 2>"$TMP/gp.err")"
if ! echo "$GP_OUT" | grep -q "gpuarray_gemm checksum = 256.0"; then
  echo "FAIL: in-process gpuArray gemm wrong/missing result"
  echo "  stdout: $(echo "$GP_OUT" | tr '\n' '|')"
  fail_n=$((fail_n+1))
elif ! grep -q "gemm dispatched to cuda" "$TMP/gp.err"; then
  echo "FAIL: gpuArray gemm did NOT dispatch to the CUDA device (ran on CPU?)"
  sed 's/^/  /' "$TMP/gp.err" | grep -i "matlab_gpu" | head
  fail_n=$((fail_n+1))
else
  echo "  gpuArray Ag*Bg ran on the CUDA device, checksum 256.0 OK"
  pass=$((pass+1))
fi

echo "cuda validation: passed $pass, failed $fail_n"
[[ $fail_n -eq 0 ]] || exit 1
echo "PASS (cuda validation)"
