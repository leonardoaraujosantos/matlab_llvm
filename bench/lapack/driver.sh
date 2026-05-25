#!/usr/bin/env bash
# bench/lapack/driver.sh <tag>
# Runs every kernel × size × impl, writes results/<tag>.json.
#
# The benchmark intentionally avoids the cocotb / plot / sym side
# dependencies that test/Run carries — pure linalg, three binaries
# (.m → matlabc → clang link, NumPy, pure Python).

set -u

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "usage: $0 <tag>" >&2
  echo "  e.g. $0 baseline_pre_lapack" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BDIR="$ROOT/bench/lapack"
MATLABC="${MATLABC:-$ROOT/build/matlabc}"
if [[ ! -x "$MATLABC" ]]; then
  echo "error: matlabc binary not found at $MATLABC" >&2
  echo "  build first: cmake --build $ROOT/build --target matlabc" >&2
  exit 2
fi

if [[ -z "${CLANG:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/clang ]]; then
    CLANG=/opt/homebrew/opt/llvm/bin/clang
  else
    CLANG=clang
  fi
fi
CXX="${CXX:-${CLANG}++}"
CXXSTD="${CXXSTD:--std=c++20}"

# Tier 4 (acceleration_roadmap §5) — let clang autovec to the host's
# native vector ISA (NEON / AVX2 / AVX-512 / Apple AMX).  Trade-off:
# the resulting binary is not portable across CPU families.  This is
# fine for benches because they're disposable.  Toggle off via
# `MARCH_NATIVE=0` to capture the pre-Tier-4 baseline.
MARCH_NATIVE="${MARCH_NATIVE:-1}"
MARCH_FLAG=""
if [[ "$MARCH_NATIVE" == "1" ]]; then
  MARCH_FLAG="-march=native"
  echo "SIMD tuning: -march=native" >&2
else
  echo "SIMD tuning: OFF (generic target)" >&2
fi

# Pin BLAS to single-threaded so per-implementation comparisons are
# fair (NumPy on macOS uses Accelerate, NumPy on Linux uses OpenBLAS;
# both spawn pools that would skew the comparison vs. our matlab_llvm
# binary that runs single-threaded today).
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p "$BDIR/results"
RESULTS="$BDIR/results/$TAG.json"

# --- Precompile the runtime once -------------------------------------------
OBJDIR="$(mktemp -d -t mlc-bench.XXXXXX)"
trap 'rm -rf "$OBJDIR"' EXIT
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
  "$ROOT/runtime/toolbox/stateflow/runtime_mstateflow.cpp"
  "$ROOT/runtime/gpu/runtime_gpu.cpp"
  "$ROOT/runtime/toolbox/gpu/runtime_gpu_helpers.cpp"
)
# Build flag: WITH_BLAS toggles LAPACK acceleration in the runtime.
# Default ON on macOS (Accelerate framework is preinstalled); explicit
# `WITH_BLAS=0` env disables for the baseline-pre-lapack measurement.
WITH_BLAS="${WITH_BLAS:-1}"
BLAS_DEFINE=""
BLAS_LINK=""
if [[ "$WITH_BLAS" == "1" ]]; then
  BLAS_DEFINE="-DMATLAB_LLVM_WITH_BLAS"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    BLAS_LINK="-framework Accelerate"
  else
    BLAS_LINK="-lblas -llapack"
  fi
  echo "BLAS dispatch: ON ($BLAS_LINK)" >&2
else
  echo "BLAS dispatch: OFF (naive O(N^3) only)" >&2
fi

# Phase 4 of lapack_roadmap §4 — Metal MPS dispatch.  Adds the Metal
# Obj-C++ TU to the runtime so `gpucoder.gemm(A,B)` lights up.
# Auto-on for macOS; off elsewhere.  Disable with `WITH_METAL=0`.
WITH_METAL="${WITH_METAL:-}"
METAL_SRCS=()
METAL_LINK=""
if [[ -z "$WITH_METAL" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then WITH_METAL=1; else WITH_METAL=0; fi
fi
if [[ "$WITH_METAL" == "1" && "$(uname -s)" == "Darwin" ]]; then
  METAL_SRCS=( "$ROOT/runtime/gpu/metal/runtime_gpu_metal.mm" )
  METAL_LINK="-framework Metal -framework MetalPerformanceShaders -framework Foundation"
  # Apple's newer Obj-C `msgsend selector stubs` are linker-resolved
  # by the system clang's libobjc; Homebrew clang's libtool can't find
  # them.  Force the link step through `/usr/bin/clang++` when Metal
  # is in the link line.
  if [[ -x /usr/bin/clang++ ]]; then
    LINK_CXX=/usr/bin/clang++
  else
    LINK_CXX="$CXX"
  fi
  echo "Metal GPU lane: ON ($METAL_LINK; linker=$LINK_CXX)" >&2
else
  LINK_CXX="$CXX"
  echo "Metal GPU lane: OFF" >&2
fi

echo "Precompiling runtime ($(uname -s)/$(uname -m), $(${CXX} --version | head -1))..." >&2
RUNTIME_OBJS=()
for src in "${RUNTIME_SRCS[@]}"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD -O3 $MARCH_FLAG $BLAS_DEFINE -I"$ROOT/runtime" -c "$src" -o "$obj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
done
# Metal Obj-C++ TU — same -O3 + -march=native, but compiled via the
# system clang (Apple SDK) so the new objc_msgSend selector-stub ABI
# lines up with Apple's libobjc at link time.
for msrc in "${METAL_SRCS[@]}"; do
  mobj="$OBJDIR/$(basename "${msrc%.mm}").o"
  MM_CXX="${LINK_CXX:-/usr/bin/clang++}"
  if ! "$MM_CXX" $CXXSTD -O3 $MARCH_FLAG -I"$ROOT/runtime" -c "$msrc" -o "$mobj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile metal TU $msrc" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$mobj" )
done

# --- Build the matlabc binary for one .m file ---
# Substitutes __BENCH_N__ with the size before compiling, so the .m can
# use a literal N without depending on a (not-yet-shipped) getenv builtin.
build_matlabc_bench() {
  local mfile="$1"; local N="$2"; local outbin="$3"
  local tmpm tmpll
  tmpm="$(mktemp -t mlc-bench.XXXXXX).m"
  sed "s/__BENCH_N__/$N/g" "$mfile" > "$tmpm"
  tmpll="$(mktemp -t mlc-bench.XXXXXX).ll"
  if ! "$MATLABC" -emit-llvm "$tmpm" > "$tmpll" 2>"$tmpll.err"; then
    echo "FAIL build: matlabc -emit-llvm $mfile (N=$N)" >&2
    sed 's/^/  /' "$tmpll.err" >&2 | head -8
    rm -f "$tmpm" "$tmpll" "$tmpll.err"
    return 1
  fi
  if ! "$LINK_CXX" $CXXSTD -O3 $MARCH_FLAG -I"$ROOT/runtime" \
        "$tmpll" "${RUNTIME_OBJS[@]}" $BLAS_LINK $METAL_LINK -o "$outbin" 2>"$OBJDIR/link.err"; then
    echo "FAIL build: clang link $mfile (N=$N)" >&2
    sed 's/^/  /' "$OBJDIR/link.err" >&2 | head -8
    rm -f "$tmpm" "$tmpll" "$tmpll.err"
    return 1
  fi
  rm -f "$tmpm" "$tmpll" "$tmpll.err"
}

# --- Extract the `best=X.YZ s` token from a run's stdout ---
extract_best() {
  awk '/best=/ { for (i=1;i<=NF;i++) if ($i ~ /^best=/) { sub(/^best=/, "", $i); sub(/s$/, "", $i); print $i; exit } }'
}

# --- Run one (kernel, N, impl) tuple, return the best time in seconds ---
run_one() {
  local kernel="$1"; local N="$2"; local impl="$3"
  export BENCH_N="$N"
  case "$impl" in
    matlab_llvm)
      local mfile="$BDIR/bench_${kernel}.m"
      [[ -e "$mfile" ]] || { echo "skip"; return; }
      # gpu_gemm bench needs the active backend env var; for the
      # Metal lane we pass MATLAB_GPU_TARGET=metal at run time.
      local extra_env=""
      if [[ "$kernel" == "gpu_gemm" && "$WITH_METAL" == "1" ]]; then
        extra_env="MATLAB_GPU_TARGET=metal"
      fi
      local bin
      bin="$(mktemp -t mlc-bench.XXXXXX)"
      if ! build_matlabc_bench "$mfile" "$N" "$bin" >&2; then
        echo "fail"
        rm -f "$bin"
        return
      fi
      local out
      out="$(env $extra_env "$bin" 2>&1)"
      rm -f "$bin"
      echo "$out" | extract_best
      ;;
    numpy)
      # gpu_gemm has no NumPy equivalent (NumPy is CPU); skip.
      [[ "$kernel" == "gpu_gemm" ]] && { echo "skip"; return; }
      export BENCH_KERNEL="$kernel"
      python3 "$BDIR/bench_numpy.py" 2>&1 | extract_best
      ;;
    pure_python)
      # Pure-Python comparisons only make sense for kernels where the
      # algorithm is genuinely scalar (matmul triple-loop, mandelbrot
      # scalar inner loop). The LAPACK kernels (lu/qr/svd/eig/chol/inv)
      # are dispatched through library calls in NumPy too — a pure-
      # Python implementation would take minutes and tell us nothing
      # about the BLAS/LAPACK story.
      case "$kernel" in
        matmul)
          [[ "$N" == "1000" ]] && { echo "skip"; return; }
          python3 "$BDIR/bench_matmul_pure.py" 2>&1 | extract_best
          ;;
        mandelbrot)
          [[ "$N" == "1000" ]] && { echo "skip"; return; }
          python3 "$BDIR/bench_mandelbrot_pure.py" 2>&1 | extract_best
          ;;
        *)
          echo "skip"
          return
          ;;
      esac
      ;;
  esac
}

# --- Sizes per kernel ----------------------------------------------------
# All kernels at N=100, 300, 1000. Pure Python skipped at N=1000 (see above).
# Tier 7 (acceleration_roadmap §8) — let CI subset the (kernel × size ×
# impl) matrix by env var.  The full local sweep stays the default;
# `BENCH_KERNELS="matmul mandelbrot" BENCH_SIZES=300 BENCH_IMPLS="matlab_llvm numpy" driver.sh ci`
# is what the perf-bench CI lane runs (skips pure-python entirely and
# gpu_gemm on Linux, where Metal isn't available).
if [[ -n "${BENCH_KERNELS:-}" ]]; then
  IFS=' ' read -r -a KERNELS <<< "$BENCH_KERNELS"
else
  KERNELS=( matmul solve lu qr chol inv eig svd mandelbrot gpu_gemm )
fi
if [[ -n "${BENCH_SIZES:-}" ]]; then
  IFS=' ' read -r -a SIZES <<< "$BENCH_SIZES"
else
  SIZES=( 100 300 1000 )
fi
if [[ -n "${BENCH_IMPLS:-}" ]]; then
  IFS=' ' read -r -a IMPLS <<< "$BENCH_IMPLS"
else
  IMPLS=( matlab_llvm numpy pure_python )
fi

# --- Driver loop ----------------------------------------------------------
{
  printf '{\n  "tag": "%s",\n' "$TAG"
  printf '  "host": {"os": "%s", "arch": "%s", "matlabc": "%s"},\n' \
    "$(uname -s)" "$(uname -m)" "$(basename "$MATLABC")"
  printf '  "results": [\n'
  first=1
  for kernel in "${KERNELS[@]}"; do
    for N in "${SIZES[@]}"; do
      for impl in "${IMPLS[@]}"; do
        echo "  $kernel  N=$N  $impl ..." >&2
        t="$(run_one "$kernel" "$N" "$impl")"
        if [[ -z "$t" ]]; then t="null"; fi
        case "$t" in
          skip|fail|null) val="null" ;;
          *)              val="$t" ;;
        esac
        if [[ $first -eq 0 ]]; then printf ',\n'; fi
        printf '    {"kernel": "%s", "N": %d, "impl": "%s", "seconds": %s}' \
          "$kernel" "$N" "$impl" "$val"
        first=0
      done
    done
  done
  printf '\n  ]\n}\n'
} > "$RESULTS"

echo "wrote $RESULTS" >&2
