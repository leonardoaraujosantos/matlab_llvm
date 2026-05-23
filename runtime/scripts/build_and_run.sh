#!/usr/bin/env bash
# Compile a .m source into an executable using matlabc + clang + the
# matlab runtime shim, then run it.
#
# Usage: build_and_run.sh <input.m> [output-name]
#
# Environment:
#   MATLABC      path to the matlabc binary
#                (default: build-sym/matlabc if it exists, else build/matlabc)
#   CLANG        path to clang
#                (default: /opt/homebrew/opt/llvm/bin/clang)
#   SYMPP_PREFIX SymPP install prefix
#                (default: /tmp/sympp_install if it exists, else /opt/homebrew)
#
# Symbolic Math Toolbox detection:
#   The script greps the input for `syms`/`sym(`/`str2sym`/etc. If the
#   program uses sym, it falls back to the sym-enabled matlabc and links
#   runtime/runtime_sym.cpp + libsympp + GMP/MPFR. Without sym usage, it
#   uses the regular matlabc + the 3-file base runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Multi-TU runtime layout (post-2026-05 reorganization):
#   $RUNTIME_DIR/matlab_runtime.cpp    — core kernel
#   $RUNTIME_DIR/runtime_*.cpp         — core helpers (debug/complex/sparse)
#   $RUNTIME_DIR/toolbox/<name>/runtime_<name>.cpp
#                                      — per-toolbox runtimes
# See docs/runtime.md for the architecture reference.
#
# All TUs are unconditional: they have no external library dependencies
# (unlike the Plot / Sym lanes below) and the CMake static-library
# targets already include them unconditionally. Linking them here is
# what lets `writeVerilogA*`, `rationalfit`, `touchstoneWrite`,
# `fmincon`, `coverageGrid`, etc. resolve when matlabc-emitted .ll
# references their `_matlab_<toolbox>_*` symbols.
RUNTIME_SRCS=(
  "$RUNTIME_DIR/matlab_runtime.cpp"
  "$RUNTIME_DIR/runtime_debug.cpp"
  "$RUNTIME_DIR/runtime_complex.cpp"
  "$RUNTIME_DIR/runtime_sparse.cpp"
  "$RUNTIME_DIR/toolbox/prop/runtime_prop.cpp"
  "$RUNTIME_DIR/toolbox/comm/runtime_comm.cpp"
  "$RUNTIME_DIR/toolbox/rf/runtime_rf.cpp"
  "$RUNTIME_DIR/toolbox/pde/runtime_pde.cpp"
  "$RUNTIME_DIR/toolbox/optim/runtime_optim.cpp"
  "$RUNTIME_DIR/toolbox/stateflow/runtime_mstateflow.cpp"
)

# CLANG default: Homebrew LLVM on macOS, system clang elsewhere (Linux/CI).
# The hardcoded Homebrew path does not exist on the Linux CI runner.
if [[ -z "${CLANG:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/clang ]]; then
    CLANG=/opt/homebrew/opt/llvm/bin/clang
  else
    CLANG=clang
  fi
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <input.m> [output]" >&2
  exit 64
fi

INPUT="$1"
OUT="${2:-$(basename "${INPUT%.m}")}"

# --- Auto-detect Symbolic Math Toolbox usage --------------------------------
# Match `syms` declarators, `sym(` / `str2sym(` / `vpa(` constructors, and
# the sym_* matrix builtins. False positives (e.g. a comment containing
# `syms`) link a few extra libs but don't break the build.
USES_SYM=0
if grep -qE '\b(syms|sym\(|str2sym|vpa|vpasolve|nsolve|simplify|expand|factor|solve|dsolve|pdsolve|laplace|ilaplace|fourier|ifourier|ztrans|iztrans|taylor|limit|assume|assumeAlso|clearAssumptions|checkodesol|sym_matrix|sym_eye|sym_zeros|sym_det|sym_inv|sym_transpose|sym_trace|sym_rank|sym_eigenvals|sym_linsolve|sym_dsolve_system|sym_solve_2x2|sym_solve_3x3|dsolve_ivp|apply_ivp)\b' "$INPUT"; then
  USES_SYM=1
fi

# --- Pick the matlabc binary -----------------------------------------------
if [[ -z "${MATLABC:-}" ]]; then
  if [[ "$USES_SYM" == 1 && -x "$ROOT/build-sym/matlabc" ]]; then
    MATLABC="$ROOT/build-sym/matlabc"
  elif [[ -x "$ROOT/build/matlabc" ]]; then
    MATLABC="$ROOT/build/matlabc"
  else
    echo "error: no matlabc found at $ROOT/build/matlabc or $ROOT/build-sym/matlabc" >&2
    exit 2
  fi
fi
if [[ ! -x "$MATLABC" ]]; then
  echo "error: matlabc not found at $MATLABC" >&2
  exit 2
fi

if [[ "$USES_SYM" == 1 && "$MATLABC" != *"build-sym/"* ]]; then
  echo "warning: input uses Symbolic Math Toolbox builtins but MATLABC=$MATLABC is the no-sym build" >&2
  echo "         (re-run with MATLABC=$ROOT/build-sym/matlabc, or build that target with MATLAB_LLVM_WITH_SYM=ON)" >&2
fi

# --- Auto-detect Cairo plotting usage ---------------------------------------
# Programs that call any of the matlab_plot.h surface need the plot
# runtime under runtime/plot/ + Cairo on the link line. False positives
# (e.g. a struct field named `plot`) link a few extra .cpp files but
# don't break the build.
USES_PLOT=0
if grep -qE '\b(figure|gcf|subplot|plot|plot3|scatter|bar|stem|stairs|area|errorbar|histogram|imshow|imagesc|pcolor|contour|contourf|quiver|mesh|surf|title|xlabel|ylabel|zlabel|legend|colorbar|colormap|grid|hold|axis|xlim|ylim|xticks|yticks|xticklabels|yticklabels|xline|yline|view|yyaxis|semilogx|semilogy|loglog|saveas|savefig|print|text)\b' "$INPUT"; then
  USES_PLOT=1
fi

# §17.5 #6 — auto-detect cross-dialect composition. Any .m using
# `mflowlink_run(...)` pulls in runtime/runtime_mflowlink_call.cpp
# plus the matlab_llvm Flowchart static libs that own the loader,
# SignalFlowLowering, and MflowLinkSim.
USES_MFLOWLINK=0
if grep -qE '\bmflowlink_run\s*\(' "$INPUT"; then
  USES_MFLOWLINK=1
fi

# --- Plot runtime sources + Cairo flags -------------------------------------
PLOT_LINK_FLAGS=()
if [[ "$USES_PLOT" == 1 ]]; then
  # GUI launches on macOS (Xcode, IDE wrappers) inherit a stripped PATH
  # that doesn't include /opt/homebrew/bin where Homebrew installs
  # pkg-config / cairo. Look for pkg-config explicitly in known Homebrew
  # prefixes before falling back to PATH lookup.
  PKG_CONFIG="${PKG_CONFIG:-}"
  if [[ -z "$PKG_CONFIG" ]]; then
    if command -v pkg-config >/dev/null 2>&1; then
      PKG_CONFIG="$(command -v pkg-config)"
    else
      for _p in /opt/homebrew/bin/pkg-config /usr/local/bin/pkg-config /usr/bin/pkg-config; do
        if [[ -x "$_p" ]]; then PKG_CONFIG="$_p"; break; fi
      done
    fi
  fi
  if [[ -z "$PKG_CONFIG" || ! -x "$PKG_CONFIG" ]]; then
    echo "error: input uses plotting builtins but pkg-config not found" >&2
    echo "       brew install pkg-config (macOS) or apt-get install pkg-config" >&2
    echo "       (or set PKG_CONFIG=/full/path/to/pkg-config)" >&2
    exit 2
  fi
  # Make sure pkg-config can find Homebrew's .pc files when the GUI
  # launched us with a stripped PKG_CONFIG_PATH.
  for _hb in /opt/homebrew /usr/local; do
    if [[ -d "$_hb/lib/pkgconfig" ]]; then
      export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+$PKG_CONFIG_PATH:}$_hb/lib/pkgconfig"
    fi
  done
  if ! "$PKG_CONFIG" --exists cairo cairo-svg cairo-pdf; then
    echo "error: cairo / cairo-svg / cairo-pdf not discoverable via $PKG_CONFIG" >&2
    echo "       brew install cairo (macOS) or apt-get install libcairo2-dev" >&2
    echo "       PKG_CONFIG_PATH=$PKG_CONFIG_PATH" >&2
    exit 2
  fi
  RUNTIME_SRCS+=(
    "$RUNTIME_DIR/plot/c_api.cpp"
    "$RUNTIME_DIR/plot/figure.cpp"
    "$RUNTIME_DIR/plot/cairo_render.cpp"
    "$RUNTIME_DIR/plot/colormap.cpp"
    "$RUNTIME_DIR/plot/contour.cpp"
  )
  # shellcheck disable=SC2207
  PLOT_LINK_FLAGS=(
    -DMATLAB_LLVM_WITH_PLOT=1
    $("$PKG_CONFIG" --cflags cairo cairo-svg cairo-pdf)
    $("$PKG_CONFIG" --libs   cairo cairo-svg cairo-pdf)
  )
fi

# --- SymPP discovery + extra link line for sym programs --------------------
SYM_LINK_FLAGS=()
if [[ "$USES_SYM" == 1 ]]; then
  SYMPP_PREFIX="${SYMPP_PREFIX:-}"
  if [[ -z "$SYMPP_PREFIX" ]]; then
    for _p in "/tmp/sympp_install" "/opt/homebrew"; do
      if [[ -e "$_p/include/sympp/sympp.hpp" ]]; then
        SYMPP_PREFIX="$_p"; break
      fi
    done
  fi
  if [[ -z "$SYMPP_PREFIX" ]]; then
    echo "error: SymPP install not found (tried /tmp/sympp_install, /opt/homebrew)" >&2
    echo "       set SYMPP_PREFIX to override" >&2
    exit 2
  fi
  RUNTIME_SRCS+=("$RUNTIME_DIR/toolbox/sym/runtime_sym.cpp")
  SYM_LINK_FLAGS=(
    -DMATLAB_LLVM_WITH_SYM=1
    "-I$SYMPP_PREFIX/include"
    "-L$SYMPP_PREFIX/lib" -lsympp
    -I/opt/homebrew/include
    -L/opt/homebrew/lib -lgmp -lgmpxx -lmpfr
    "-Wl,-rpath,$SYMPP_PREFIX/lib"
  )
fi

# §17.5 #6 — mflowlink_run requires the Flowchart static libs that
# own loader / SignalFlowLowering / MflowLinkSim. The matlab_llvm
# build directory must be discoverable (we walk up from $MATLABC).
MFLOWLINK_LINK_FLAGS=()
if [[ "$USES_MFLOWLINK" == 1 ]]; then
  MATLABC_BUILD_DIR="$(cd "$(dirname "$MATLABC")" && pwd)"
  if [[ ! -f "$MATLABC_BUILD_DIR/libMatlabFlowchart.a" ]]; then
    echo "error: cannot find Flowchart static libs alongside $MATLABC" >&2
    echo "       (looked in $MATLABC_BUILD_DIR)" >&2
    exit 2
  fi
  RUNTIME_SRCS+=("$RUNTIME_DIR/mflowlink/runtime_mflowlink_call.cpp")
  MFLOWLINK_LINK_FLAGS=(
    -I"$ROOT/include"
    "$MATLABC_BUILD_DIR/libMatlabFlowchart.a"
    "$MATLABC_BUILD_DIR/libMatlabParse.a"
    "$MATLABC_BUILD_DIR/libMatlabLex.a"
    "$MATLABC_BUILD_DIR/libMatlabAST.a"
    "$MATLABC_BUILD_DIR/libMatlabBasic.a"
  )
fi

# --- Compile + link --------------------------------------------------------
TMP="$(mktemp -t matlabc.XXXXXX).ll"
trap 'rm -f "$TMP"' EXIT

# Runtime is C++ since Phase 3 of docs/port_runtime_2_cpp.md. Drive
# the link line with clang++ so the .cpp gets compiled as C++; the
# matlabc-emitted .ll is still C-compatible.
"$MATLABC" -emit-llvm "$INPUT" > "$TMP"
# Note the `${arr[@]+"${arr[@]}"}` dance: macOS ships bash 3.2 which
# treats an empty array as unset under `set -u`, so a bare expansion of
# SYM_LINK_FLAGS would error out for non-sym programs (the common case).
"${CLANG}++" -std=c++20 -Wno-override-module \
  "$TMP" "${RUNTIME_SRCS[@]}" \
  -I"$RUNTIME_DIR" \
  ${PLOT_LINK_FLAGS[@]+"${PLOT_LINK_FLAGS[@]}"} \
  ${SYM_LINK_FLAGS[@]+"${SYM_LINK_FLAGS[@]}"} \
  ${MFLOWLINK_LINK_FLAGS[@]+"${MFLOWLINK_LINK_FLAGS[@]}"} \
  -o "$OUT"
echo "built $OUT"
