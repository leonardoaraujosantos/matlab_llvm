#!/usr/bin/env bash
# Examples sweep — compile/link/run every in-scope examples/**/*.m through the
# canonical AOT (LLVM) execute path and classify the outcome.
#
# This is the automated form of the manual sweep documented in
# docs/examples_status_report.md.  It is wired into the *nightly* CI lane (it
# runs on a schedule regardless of code changes); it is NOT a per-PR gate.
#
# The recipe mirrors test/Run/run_tests.sh exactly: the runtime is compiled once
# into objects up front, then each example is `matlabc -emit-llvm` -> clang++
# linked against those objects -> run (from the file's own directory, with a
# timeout).  Examples carry no .stdout golden, so "OK" means compile + link +
# run with exit 0 — we verify the example doesn't break the toolchain, not its
# numerics (those are covered by the per-toolbox test/Run/*.m lanes).
#
# Outcomes per example:
#   OK       emit + link + run, exit 0
#   EMIT     matlabc -emit-llvm errored (frontend / lowering gap)
#   LINK     clang++ link failed (undefined symbols)
#   RUNTIME  ran but exited non-zero
#   TIMEOUT  ran past the time limit
#   SKIP     out of LLVM-execute scope (see SKIP rules below) — never a failure
#
# Regression gate: every non-SKIP failure (EMIT/LINK/RUNTIME/TIMEOUT) is checked
# against the committed baseline test/Examples/known_failures.txt.  A failure NOT
# in the baseline is a REGRESSION and makes the script exit non-zero.  An entry
# in the baseline that now passes is reported as STALE (prune it) but does not
# fail the run.
#
# Usage:
#   run_sweep.sh <path-to-matlabc>                  # sweep + gate on regressions
#   run_sweep.sh <path-to-matlabc> --update-baseline  # rewrite known_failures.txt
#
# Portable to bash 3.2 (macOS) — no associative arrays / mapfile; set logic is
# done with sorted temp files + comm.
set -u

MATLABC="${1:-}"
MODE="${2:-gate}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc> [--update-baseline]" >&2
  exit 2
fi
case "$MODE" in
  --update-baseline) MODE=update ;;
  gate|"")           MODE=gate ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EXDIR="$ROOT/examples"
BASELINE="$HERE/known_failures.txt"
TIMEOUT_S="${EXAMPLES_TIMEOUT:-15}"

# --- toolchain (same defaults as test/Run/run_tests.sh) ---------------------
if [[ -z "${CLANG:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/clang ]]; then
    CLANG=/opt/homebrew/opt/llvm/bin/clang
  else
    CLANG=clang
  fi
fi
CXX="${CXX:-${CLANG}++}"
CXXSTD="${CXXSTD:--std=c++20}"   # runtime is a C++20 project (see run_tests.sh)

# `timeout` (GNU) on Linux, `gtimeout` (coreutils) on macOS, else no limit.
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"; fi

# --- runtime TU set — keep in sync with test/Run/run_tests.sh ---------------
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
  "$ROOT/runtime/toolbox/stateflow/runtime_mstateflow.cpp"
)

WORK="$(mktemp -d -t mlc-sweep.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Compiling runtime objects (once)…"
RUNTIME_OBJS=()
for src in "${RUNTIME_SRCS[@]}"; do
  obj="$WORK/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD -DMATLAB_LLVM_WITH_PLOT=1 -I"$ROOT/runtime" -c "$src" -o "$obj" 2>"$WORK/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$WORK/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
done

# Cairo plot runtime (precompiled once iff cairo is available).
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
    obj="$WORK/plot_$src.o"
    if ! "$CXX" -DMATLAB_LLVM_WITH_PLOT=1 -I"$ROOT/runtime" \
           "${_plot_cflags[@]}" -c "$ROOT/runtime/plot/$src.cpp" -o "$obj" 2>/dev/null; then
      PLOT_OK=0; break
    fi
    PLOT_OBJS+=( "$obj" )
  done
fi

# Symbolic Math Toolbox (SymPP) — detected the same way as run_tests.sh's
# .requires-sym path.  Present => sym examples are linked against it; absent =>
# sym examples are SKIPped (out of this environment's scope, never a failure).
SYM_OK=0
SYM_OBJS=()
SYM_LIBS=()
symo="$ROOT/build/CMakeFiles/matlabc.dir/runtime/toolbox/sym/runtime_sym.cpp.o"
symdir="${SYMPP_DIR:-}"
if [[ -z "$symdir" && -e "$ROOT/build/CMakeCache.txt" ]]; then
  symdir="$(sed -n 's/^SymPP_DIR[^=]*=//p' "$ROOT/build/CMakeCache.txt" | head -1)"
fi
symlibdir=""
for cand in \
    "$symdir/src" "$symdir/lib" "$symdir/lib64" \
    "$symdir/../.." "$symdir/../../lib" "$symdir/../../lib64" \
    "${SYMPP_PREFIX:-/tmp/sympp_install}/lib" \
    "${SYMPP_PREFIX:-/tmp/sympp_install}/lib64"; do
  if compgen -G "$cand/libsympp.*" >/dev/null 2>&1; then
    symlibdir="$(cd "$cand" && pwd)"; break
  fi
done
if [[ -e "$symo" && -n "$symlibdir" ]]; then
  SYM_OK=1
  SYM_OBJS=( "$symo" )
  SYM_LIBS=( -L"$symlibdir" -lsympp -Wl,-rpath,"$symlibdir" )
  for gl in /opt/homebrew/lib /usr/local/lib; do
    [[ -d "$gl" ]] && SYM_LIBS+=( -L"$gl" -Wl,-rpath,"$gl" )
  done
  SYM_LIBS+=( -lgmp -lmpfr )
fi

echo "==> plot runtime: $([[ $PLOT_OK == 1 ]] && echo available || echo MISSING)   symbolic (SymPP): $([[ $SYM_OK == 1 ]] && echo available || echo MISSING)"
echo "==> timeout: ${TIMEOUT_BIN:-none} (${TIMEOUT_S}s)"
echo

# --- per-example sweep ------------------------------------------------------
# Status lines accumulate into all.tsv as "STATUS<TAB>relpath".
ALL="$WORK/all.tsv"; : >"$ALL"

# An example is in SKIP scope if its relative path matches one of these
# prefixes (SystemVerilog / cocotb / flowchart-dialect targets, not standalone
# LLVM-execute programs — see docs/examples_status_report.md §A and §F).
skip_scope() {
  case "$1" in
    hdl/*)        return 0 ;;  # SV / cocotb modules + testbenches + synth wrappers
    mflow/*)      return 0 ;;  # custom-block fragments, run via .mflow tooling
    mflowlink/*)  return 0 ;;  # cross-dialect fragments, run via mflowlink_run
    stateflow/*)  return 0 ;;  # state-chart fragments
    *)            return 1 ;;
  esac
}

# Heuristic: does the example need the Symbolic Math Toolbox?
needs_sym() { grep -qE '(^|[^[:alnum:]_])(syms|sym)[[:space:]]*[(]|^[[:space:]]*syms[[:space:]]' "$1"; }

count_total=0
while IFS= read -r m; do
  rel="${m#$EXDIR/}"
  base="${rel%.m}"
  count_total=$((count_total+1))

  if skip_scope "$rel"; then
    printf 'SKIP\t%s\n' "$rel" >>"$ALL"; continue
  fi
  if needs_sym "$m" && [[ $SYM_OK != 1 ]]; then
    # Symbolic example but no linkable SymPP in this environment — out of
    # scope here, not a failure.  Tracked separately so the report makes it
    # obvious whether the sym examples actually ran (they should, when SymPP
    # is built — see the nightly CI job).
    printf 'SKIPSYM\t%s\n' "$rel" >>"$ALL"; continue
  fi

  tmpll="$WORK/x.ll"; tmpbin="$WORK/x.out"; rm -f "$tmpll" "$tmpbin"

  if ! "$MATLABC" -emit-llvm "$m" >"$tmpll" 2>"$WORK/emit.err"; then
    printf 'EMIT\t%s\n' "$rel" >>"$ALL"; continue
  fi

  link_objs=( "${RUNTIME_OBJS[@]}" )
  link_libs=()
  if [[ $PLOT_OK == 1 ]]; then
    link_objs+=( "${PLOT_OBJS[@]}" )
    link_libs+=( "${PLOT_LIBS[@]}" )
  fi
  if needs_sym "$m" && [[ $SYM_OK == 1 ]]; then
    link_objs+=( "${SYM_OBJS[@]}" )
    link_libs+=( "${SYM_LIBS[@]}" )
  fi

  if ! "$CXX" -DMATLAB_LLVM_WITH_PLOT=1 -Wno-override-module "$tmpll" \
              "${link_objs[@]}" -I"$ROOT/runtime" \
              ${link_libs[@]+"${link_libs[@]}"} \
              -o "$tmpbin" 2>"$WORK/link.err"; then
    printf 'LINK\t%s\n' "$rel" >>"$ALL"; continue
  fi

  # Run from the example's own directory so relative fixture paths resolve.
  exdir="$(dirname "$m")"
  if [[ -n "$TIMEOUT_BIN" ]]; then
    ( cd "$exdir" && "$TIMEOUT_BIN" "$TIMEOUT_S" "$tmpbin" ) >/dev/null 2>&1
  else
    ( cd "$exdir" && "$tmpbin" ) >/dev/null 2>&1
  fi
  rc=$?
  if [[ $rc -eq 124 ]]; then
    printf 'TIMEOUT\t%s\n' "$rel" >>"$ALL"
  elif [[ $rc -ne 0 ]]; then
    printf 'RUNTIME\t%s\n' "$rel" >>"$ALL"
  else
    printf 'OK\t%s\n' "$rel" >>"$ALL"
  fi
done < <(find "$EXDIR" -name '*.m' | sort)

# --- coverage assertion -----------------------------------------------------
# Every .m under examples/ must produce exactly one status line; otherwise an
# example was silently dropped (e.g. a code path that 'continue'd without
# recording a result).  Fail loudly so "we run all the examples" stays true.
n_lines=$(grep -c . "$ALL" 2>/dev/null || echo 0)
n_found=$(find "$EXDIR" -name '*.m' | wc -l | tr -d ' ')
if [[ "$n_lines" -ne "$n_found" || "$count_total" -ne "$n_found" ]]; then
  echo "FATAL: coverage mismatch — found $n_found .m files, swept $count_total, recorded $n_lines result lines." >&2
  exit 2
fi

# --- tally ------------------------------------------------------------------
n_ok=$(awk -F'\t' '$1=="OK"{c++} END{print c+0}' "$ALL")
n_emit=$(awk -F'\t' '$1=="EMIT"{c++} END{print c+0}' "$ALL")
n_link=$(awk -F'\t' '$1=="LINK"{c++} END{print c+0}' "$ALL")
n_rt=$(awk -F'\t' '$1=="RUNTIME"{c++} END{print c+0}' "$ALL")
n_to=$(awk -F'\t' '$1=="TIMEOUT"{c++} END{print c+0}' "$ALL")
n_skip=$(awk -F'\t' '$1=="SKIP"{c++} END{print c+0}' "$ALL")
n_skipsym=$(awk -F'\t' '$1=="SKIPSYM"{c++} END{print c+0}' "$ALL")
n_inscope=$(( n_ok + n_emit + n_link + n_rt + n_to ))

# Current failures (non-OK, non-SKIP*), sorted.
awk -F'\t' '$1!="OK" && $1!="SKIP" && $1!="SKIPSYM"{print $2}' "$ALL" | sort >"$WORK/fail.txt"

# Baseline -> bare paths: strip inline/full-line comments (the writer aligns a
# "# STATUS" note onto each entry for readers) and trailing whitespace, drop
# blanks, sort.
if [[ -e "$BASELINE" ]]; then
  sed -E 's/#.*$//; s/[[:space:]]+$//' "$BASELINE" | grep -vE '^[[:space:]]*$' | sort -u >"$WORK/base.txt"
else
  : >"$WORK/base.txt"
fi

comm -23 "$WORK/fail.txt" "$WORK/base.txt" >"$WORK/regressions.txt"  # fail & !baseline
comm -13 "$WORK/fail.txt" "$WORK/base.txt" >"$WORK/stale.txt"        # baseline & !fail
n_reg=$(wc -l <"$WORK/regressions.txt" | tr -d ' ')
n_stale=$(wc -l <"$WORK/stale.txt" | tr -d ' ')

# --- --update-baseline ------------------------------------------------------
if [[ "$MODE" == update ]]; then
  {
    echo "# Known-failing examples for the nightly sweep (test/Examples/run_sweep.sh)."
    echo "# Generated by --update-baseline; each line is a path under examples/."
    echo "# A failure NOT listed here is treated as a regression (fails the lane)."
    echo "# Out-of-scope dirs (hdl/ mflow/ mflowlink/ stateflow/) are SKIPped, not listed."
    echo "# Regenerated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    awk -F'\t' '$1!="OK" && $1!="SKIP" && $1!="SKIPSYM"{printf "%-50s # %s\n", $2, $1}' "$ALL" | sort
  } >"$BASELINE"
  echo "Wrote baseline: $BASELINE ($(grep -cvE '^[[:space:]]*(#|$)' "$BASELINE") entries)"
fi

# --- report -----------------------------------------------------------------
print_report() {
  echo "## Examples sweep — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Total examples: $n_found  =  $n_inscope in-scope (run)  +  $(( n_skip + n_skipsym )) skipped"
  echo
  echo "| Result | Count |"
  echo "|---|---|"
  echo "| **OK** (compile + link + run, exit 0) | **$n_ok / $n_inscope** |"
  echo "| EMIT (frontend / lowering error) | $n_emit |"
  echo "| LINK (undefined symbols) | $n_link |"
  echo "| RUNTIME (non-zero exit) | $n_rt |"
  echo "| TIMEOUT | $n_to |"
  echo "| SKIP (HDL / flowchart — own CI lanes) | $n_skip |"
  echo "| SKIP (symbolic — no linkable SymPP here) | $n_skipsym |"
  echo
  echo "Baseline known-failures: $(wc -l <"$WORK/base.txt" | tr -d ' ')   ·   Regressions: $n_reg   ·   Stale baseline entries: $n_stale"
  if [[ "$n_reg" -gt 0 ]]; then
    echo
    echo "### ❌ Regressions (failing, not in baseline)"
    echo '```'
    while IFS= read -r r; do
      [[ -z "$r" ]] && continue
      st=$(awk -F'\t' -v p="$r" '$2==p{print $1}' "$ALL")
      echo "$st  $r"
    done <"$WORK/regressions.txt"
    echo '```'
  fi
  if [[ "$n_stale" -gt 0 ]]; then
    echo
    echo "### ⚠️ Stale baseline entries (now passing — prune from known_failures.txt)"
    echo '```'
    cat "$WORK/stale.txt"
    echo '```'
  fi
}

print_report
# Mirror the report into the GitHub Actions job summary when present.
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  print_report >>"$GITHUB_STEP_SUMMARY"
fi

if [[ "$MODE" == gate && "$n_reg" -gt 0 ]]; then
  echo
  echo "FAIL: $n_reg example regression(s)."
  exit 1
fi
echo
echo "OK: no example regressions."
exit 0
