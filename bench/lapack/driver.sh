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

echo "Precompiling runtime ($(uname -s)/$(uname -m), $(${CXX} --version | head -1))..." >&2
RUNTIME_OBJS=()
for src in "${RUNTIME_SRCS[@]}"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD -O3 $BLAS_DEFINE -I"$ROOT/runtime" -c "$src" -o "$obj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
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
  if ! "$CXX" $CXXSTD -O3 -I"$ROOT/runtime" \
        "$tmpll" "${RUNTIME_OBJS[@]}" $BLAS_LINK -o "$outbin" 2>"$OBJDIR/link.err"; then
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
      local bin
      bin="$(mktemp -t mlc-bench.XXXXXX)"
      if ! build_matlabc_bench "$mfile" "$N" "$bin" >&2; then
        echo "fail"
        rm -f "$bin"
        return
      fi
      local out
      out="$("$bin" 2>&1)"
      rm -f "$bin"
      echo "$out" | extract_best
      ;;
    numpy)
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
KERNELS=( matmul solve lu qr chol inv eig svd mandelbrot )
SIZES=( 100 300 1000 )
IMPLS=( matlab_llvm numpy pure_python )

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
