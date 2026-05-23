#!/usr/bin/env bash
# Build-and-run tests. For each .m in this directory, compiles with matlabc
# + clang + the matlab runtime, runs the executable, and compares stdout to
# the matching .stdout file. Failure if stdout differs or exit is non-zero.
#
# The runtime is compiled ONCE into objects up front and linked per fixture
# (it used to be recompiled from source for every fixture — ~15 large TUs each
# time — which dominated the lane's wall time; this is a ~10-40x speedup).
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# CLANG default: Homebrew LLVM on macOS, system clang elsewhere (Linux/CI).
# The hardcoded Homebrew path does not exist on the Linux CI runner.
if [[ -z "${CLANG:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/clang ]]; then
    CLANG=/opt/homebrew/opt/llvm/bin/clang
  else
    CLANG=clang
  fi
fi
# The runtime is a C++20 project (CMAKE_CXX_STANDARD 20).  Compile the
# runtime TUs with the same standard the cmake build uses — under the
# compiler default (gnu++17) libstdc++ does not transitively pull in
# <string>/<cstdio>, so the runtime fails to compile on Linux.
CXXSTD="${CXXSTD:--std=c++20}"
# Runtime is C++ since Phase 3 of docs/port_runtime_2_cpp.md — drive the
# link line with clang++ so the .cpp is compiled as C++.
# Post-2026-05 reorganization: core runtime stays at runtime/ top
# level, per-toolbox runtimes live under runtime/toolbox/<name>/.
# All share private layouts via runtime_internal.h; every TU must
# appear on every link line.
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
  "$ROOT/runtime/toolbox/stateflow/runtime_mstateflow.cpp"
)
CXX="${CXX:-${CLANG}++}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

# --- Precompile the runtime once -------------------------------------------
OBJDIR="$(mktemp -d -t mlc-rt.XXXXXX)"
trap 'rm -rf "$OBJDIR"' EXIT
RUNTIME_OBJS=()
for src in "${RUNTIME_SRCS[@]}"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD -DMATLAB_LLVM_WITH_PLOT=1 -I"$ROOT/runtime" -c "$src" -o "$obj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
done

# Cairo plot runtime — precompiled once iff cairo is available; the per-fixture
# link adds these objects + libs only for `.requires-plot` tests.
PLOT_OK=0
PLOT_OBJS=()
PLOT_LIBS=()
if command -v pkg-config >/dev/null 2>&1 && \
   pkg-config --exists cairo cairo-svg cairo-pdf 2>/dev/null; then
  # shellcheck disable=SC2207
  _plot_cflags=( $(pkg-config --cflags cairo cairo-svg cairo-pdf) )
  # shellcheck disable=SC2207
  PLOT_LIBS=( $(pkg-config --libs cairo cairo-svg cairo-pdf) )
  PLOT_OK=1
  for src in c_api cairo_render colormap contour figure; do
    obj="$OBJDIR/plot_$src.o"
    if ! "$CXX" -DMATLAB_LLVM_WITH_PLOT=1 -I"$ROOT/runtime" \
           "${_plot_cflags[@]}" -c "$ROOT/runtime/plot/$src.cpp" -o "$obj" 2>/dev/null; then
      PLOT_OK=0; break
    fi
    PLOT_OBJS+=( "$obj" )
  done
fi

pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $m (no .stdout)"; continue; }

  tmpll="$(mktemp -t mlc.XXXXXX).ll"
  tmpbin="$(mktemp -t mlc.XXXXXX).out"

  if ! "$MATLABC" -emit-llvm "$m" > "$tmpll" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-llvm errored"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  fi
  # Per-test opt-in for the Cairo plot runtime (precompiled objects above).
  # If the marker is present but cairo isn't available, SKIP rather than fail.
  plot_objs=()
  plot_libs=()
  if [[ -e "${m%.m}.requires-plot" ]]; then
    if [[ "$PLOT_OK" != 1 ]]; then
      echo "SKIP $base (requires-plot, no cairo)"
      rm -f "$tmpll" "$tmpbin"; continue
    fi
    plot_objs=( "${PLOT_OBJS[@]}" )
    plot_libs=( "${PLOT_LIBS[@]}" )
  fi
  # Per-test opt-in for the Symbolic Math Toolbox (SymPP).  A
  # `<name>.requires-sym` marker links the prebuilt WITH_SYM runtime object
  # (build/.../runtime_sym.cpp.o — only present when the build was configured
  # with -DMATLAB_LLVM_WITH_SYM=ON, which carries the SymPP include flags) plus
  # libsympp and its GMP / MPFR deps.  SymPP_DIR is read from CMakeCache so it
  # matches how matlabc itself was built.  Missing object or lib -> SKIP.
  sym_objs=()
  sym_libs=()
  if [[ -e "${m%.m}.requires-sym" ]]; then
    symo="$ROOT/build/CMakeFiles/matlabc.dir/runtime/toolbox/sym/runtime_sym.cpp.o"
    symdir="${SYMPP_DIR:-}"
    if [[ -z "$symdir" && -e "$ROOT/build/CMakeCache.txt" ]]; then
      symdir="$(sed -n 's/^SymPP_DIR[^=]*=//p' "$ROOT/build/CMakeCache.txt" | head -1)"
    fi
    symlibdir=""
    for sub in src lib lib64; do
      if compgen -G "$symdir/$sub/libsympp.*" >/dev/null 2>&1; then
        symlibdir="$symdir/$sub"; break
      fi
    done
    if [[ ! -e "$symo" || -z "$symlibdir" ]]; then
      echo "SKIP $base (requires-sym, no SymPP build)"
      rm -f "$tmpll" "$tmpbin"; continue
    fi
    sym_objs=( "$symo" )
    sym_libs=( -L"$symlibdir" -lsympp -Wl,-rpath,"$symlibdir" )
    for gl in /opt/homebrew/lib /usr/local/lib; do
      [[ -d "$gl" ]] && sym_libs+=( -L"$gl" -Wl,-rpath,"$gl" )
    done
    sym_libs+=( -lgmp -lmpfr )
  fi
  if ! "$CXX" -DMATLAB_LLVM_WITH_PLOT=1 -Wno-override-module "$tmpll" \
              "${RUNTIME_OBJS[@]}" \
              ${plot_objs[@]+"${plot_objs[@]}"} \
              ${sym_objs[@]+"${sym_objs[@]}"} \
              -I"$ROOT/runtime" \
              ${plot_libs[@]+"${plot_libs[@]}"} \
              ${sym_libs[@]+"${sym_libs[@]}"} \
              -o "$tmpbin" 2>/dev/null; then
    echo "FAIL $base: clang link failed"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  fi
  # Run from TESTDIR so a fixture referenced by a path relative to the test
  # directory (e.g. fixtures/rf/test_amp.s2p) resolves on every platform —
  # the harness CWD differs (build/ under ctest, repo root locally).
  got="$(cd "$TESTDIR" && "$tmpbin")" || {
    echo "FAIL $base: non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  }
  # If a .sorted file exists alongside the .m, compare against the expected
  # output after sorting both sides (useful for parfor where iteration
  # order is nondeterministic).
  # Comparison is tolerance-aware (numdiff.py): numeric tokens match within
  # a relative tolerance, text exactly — absorbs last-digit libm divergence
  # between macOS (libc++) and Linux (libstdc++) in the disp-printed goldens.
  ND=(python3 "$ROOT/test/Run/numdiff.py")
  if [[ -e "${m%.m}.sorted" ]]; then
    if "${ND[@]}" <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch (sorted)"
      "${ND[@]}" <(sort "$exp") <(printf '%s\n' "$got" | sort) | sed 's/^/  /'
    fi
  elif "${ND[@]}" "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
    "${ND[@]}" "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
  rm -f "$tmpll" "$tmpbin"
done

echo "----"
echo "run passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
