#!/usr/bin/env bash
# GPU array-capture outliner CI lane (issue #24).
#
# Verifies the real outliner reached by MATLAB_GPU_OUTLINE=1:
#   1. a flat element-wise `coder.gpu.kernelfun` kernel (AXPY/map shape)
#      is OUTLINED — the emitted LLVM IR defines a standalone
#      `__gpu_kernel_N` function (the pre-lowering rewrite never emits
#      one), and the program runs bit-exact vs the default `matlab.for`
#      lane through the matlab_gpu_launch_kernel CPU dispatch;
#   2. a kernel outside the supported class (nested loops + scalar
#      temporaries — Mandelbrot) DECLINES cleanly and still compiles +
#      runs with the same result as the default lane (the flag never
#      corrupts a program it can't fully outline).
#
# Runs on any host — the CPU dispatch fallback executes the outlined
# function sequentially, so no GPU hardware is required.
#
# Usage: run_gpu_outline_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

# CLANG default: Homebrew LLVM on macOS, system clang elsewhere.
if [[ -z "${CLANG:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/clang ]]; then
    CLANG=/opt/homebrew/opt/llvm/bin/clang
  else
    CLANG=clang
  fi
fi
CXX="${CXX:-${CLANG}++}"
CXXSTD="${CXXSTD:--std=c++20}"

# Runtime TUs — the same set the main run_tests.sh links (CPU lane; the
# Metal/CUDA/OpenCL .mm/.cpp backends are intentionally excluded so the
# weak matlab_gpu_launch_* stubs route to the sequential CPU fallback).
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
  "$ROOT/runtime/runtime_sparse.cpp"
  "$ROOT/runtime/toolbox/prop/runtime_prop.cpp"
  "$ROOT/runtime/toolbox/comm/runtime_comm.cpp"
  "$ROOT/runtime/toolbox/rf/runtime_rf.cpp"
  "$ROOT/runtime/toolbox/pde/runtime_pde.cpp"
  "$ROOT/runtime/toolbox/optim/runtime_optim.cpp"
  "$ROOT/runtime/toolbox/mpc/runtime_mpc.cpp"
  "$ROOT/runtime/toolbox/ident/runtime_ident.cpp"
  "$ROOT/runtime/toolbox/gads/runtime_gads.cpp"
  "$ROOT/runtime/toolbox/stats/runtime_stats.cpp"
  "$ROOT/runtime/toolbox/images/runtime_images.cpp"
  "$ROOT/runtime/toolbox/curvefit/runtime_curvefit.cpp"
  "$ROOT/runtime/toolbox/wavelet/runtime_wavelet.cpp"
  "$ROOT/runtime/toolbox/dsp/runtime_dsp.cpp"
  "$ROOT/runtime/toolbox/finance/runtime_finance.cpp"
  "$ROOT/runtime/toolbox/econ/runtime_econ.cpp"
  "$ROOT/runtime/toolbox/fusion/runtime_fusion.cpp"
  "$ROOT/runtime/toolbox/robotics/runtime_robotics.cpp"
  "$ROOT/runtime/toolbox/navigation/runtime_navigation.cpp"
  "$ROOT/runtime/toolbox/dlnet/runtime_dlnet.cpp"
  "$ROOT/runtime/toolbox/dlnet/runtime_onnx.cpp"
  "$ROOT/runtime/toolbox/rl/runtime_rl.cpp"
  "$ROOT/runtime/toolbox/stateflow/runtime_mstateflow.cpp"
  "$ROOT/runtime/gpu/runtime_gpu.cpp"
  "$ROOT/runtime/toolbox/gpu/runtime_gpu_helpers.cpp"
)

OBJDIR="$(mktemp -d -t gpu-outline.XXXXXX)"
trap 'rm -rf "$OBJDIR"' EXIT

RUNTIME_OBJS=()
for src in "${RUNTIME_SRCS[@]}"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD -DMATLAB_LLVM_WITH_PLOT=1 -I"$ROOT/runtime" -c "$src" \
        -o "$obj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
done

# emit+link+run a fixture under an env; echo its stdout (or FAIL:<msg>).
run_lane () {
  local m="$1" env_assign="$2" tag="$3"
  local ll="$OBJDIR/$tag.ll" bin="$OBJDIR/$tag.bin"
  if ! env $env_assign "$MATLABC" -emit-llvm "$m" > "$ll" 2>"$ll.err"; then
    echo "FAIL:emit"; return
  fi
  if ! "$CXX" $CXXSTD -Wno-override-module -DMATLAB_LLVM_WITH_PLOT=1 \
        "$ll" "${RUNTIME_OBJS[@]}" -I"$ROOT/runtime" -o "$bin" \
        2>"$ll.lk"; then
    echo "FAIL:link"; return
  fi
  "$bin" 2>/dev/null || echo "FAIL:run"
}

pass=0
fail=0
check () {  # name expected actual
  if [[ "$2" == "$3" ]]; then
    echo "PASS $1"
    pass=$((pass+1))
  else
    echo "FAIL $1: expected [$2] got [$3]"
    fail=$((fail+1))
  fi
}

# --- Test 1: flat element-wise kernel is outlined + bit-exact. ---------
AXPY="$TESTDIR/gpu_outline_axpy.m"
EXP="$(cat "$TESTDIR/gpu_outline_axpy.stdout")"

# 1a. The outline lane must EMIT a real __gpu_kernel_N function.
if MATLAB_GPU_OUTLINE=1 "$MATLABC" -emit-llvm "$AXPY" 2>/dev/null \
     | grep -q "define void @__gpu_kernel"; then
  echo "PASS axpy:outlined (emits __gpu_kernel_N)"
  pass=$((pass+1))
else
  echo "FAIL axpy:outlined — no __gpu_kernel_N in MATLAB_GPU_OUTLINE=1 IR"
  fail=$((fail+1))
fi

# 1b. Both lanes run and agree with the golden output.
check "axpy:default-run"  "$EXP" "$(run_lane "$AXPY" "" axdef)"
check "axpy:outline-run"  "$EXP" "$(run_lane "$AXPY" "MATLAB_GPU_OUTLINE=1" axout)"

# --- Test 2: scalar-temp + nested-loop kernel (Mandelbrot) outlines. ---
# The canonical coder.gpu.kernelfun demo: nested for-loops, a while
# loop, scalar temporaries (cr/zr/zi/k) and an output array.  The
# outliner clones each scalar slot per-invocation (seeded from its outer
# value) and shares the array slot, so it lifts into a real
# __gpu_kernel_N and runs bit-exact vs the default lane.
MB="$TESTDIR/gpu_mandelbrot.m"
MBEXP="$(cat "$TESTDIR/gpu_mandelbrot.stdout" 2>/dev/null || echo 'mandelbrot checksum = 6336')"

# 2a. The outline lane must emit a kernel.
if MATLAB_GPU_OUTLINE=1 "$MATLABC" -emit-llvm "$MB" 2>/dev/null \
     | grep -q "define void @__gpu_kernel"; then
  echo "PASS mandelbrot:outlined (scalar temps cloned, array shared)"
  pass=$((pass+1))
else
  echo "FAIL mandelbrot:outlined — no __gpu_kernel_N in MATLAB_GPU_OUTLINE=1 IR"
  fail=$((fail+1))
fi

# 2b. Both lanes run and agree.
check "mandelbrot:default-run" "$MBEXP" "$(run_lane "$MB" "" mbdef)"
check "mandelbrot:outline-run" "$MBEXP" "$(run_lane "$MB" "MATLAB_GPU_OUTLINE=1" mbout)"

echo "----"
echo "gpu-outline: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
