#!/usr/bin/env bash
# Build-and-run tests. For each .m in this directory, compiles with matlabc
# + clang + the matlab runtime, runs the executable, and compares stdout to
# the matching .stdout file. Failure if stdout differs or exit is non-zero.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
# Runtime is C++ since Phase 3 of docs/port_runtime_2_cpp.md — drive the
# link line with clang++ so the .cpp is compiled as C++.
# Phase-2 + 2.5 split: three .cpp files share private layouts via
# runtime_internal.h. All three must appear on every link line.
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
  "$ROOT/runtime/runtime_comm.cpp"
  "$ROOT/runtime/runtime_prop.cpp"
  "$ROOT/runtime/runtime_rf.cpp"
  "$ROOT/runtime/runtime_pde.cpp"
)
CXX="${CXX:-${CLANG}++}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

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
  # Per-test opt-in for the Cairo plot runtime.  When a
  # `<name>.requires-plot` marker exists, also link runtime/plot/*.cpp
  # and the cairo pkg-config libs.  Without it the test gets a
  # plot-free link line (smaller, no Cairo dep), matching the rest of
  # the harness.  If the marker is present but pkg-config can't find
  # cairo, SKIP the test rather than fail.
  plot_srcs=()
  plot_cflags=()
  plot_libs=()
  if [[ -e "${m%.m}.requires-plot" ]]; then
    if ! command -v pkg-config >/dev/null 2>&1 || \
       ! pkg-config --exists cairo cairo-svg cairo-pdf 2>/dev/null; then
      echo "SKIP $base (requires-plot, no cairo)"
      rm -f "$tmpll" "$tmpbin"; continue
    fi
    plot_srcs=(
      "$ROOT/runtime/plot/c_api.cpp"
      "$ROOT/runtime/plot/cairo_render.cpp"
      "$ROOT/runtime/plot/colormap.cpp"
      "$ROOT/runtime/plot/contour.cpp"
      "$ROOT/runtime/plot/figure.cpp"
    )
    # shellcheck disable=SC2207
    plot_cflags=( $(pkg-config --cflags cairo cairo-svg cairo-pdf) )
    # shellcheck disable=SC2207
    plot_libs=( $(pkg-config --libs cairo cairo-svg cairo-pdf) )
  fi
  if ! "$CXX" -DMATLAB_LLVM_WITH_PLOT=1 -Wno-override-module "$tmpll" \
              "${RUNTIME_SRCS[@]}" \
              ${plot_srcs[@]+"${plot_srcs[@]}"} \
              -I"$ROOT/runtime" \
              ${plot_cflags[@]+"${plot_cflags[@]}"} \
              ${plot_libs[@]+"${plot_libs[@]}"} \
              -o "$tmpbin" 2>/dev/null; then
    echo "FAIL $base: clang link failed"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  fi
  got="$("$tmpbin")" || {
    echo "FAIL $base: non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpll" "$tmpbin"; continue
  }
  # If a .sorted file exists alongside the .m, compare against the expected
  # output after sorting both sides (useful for parfor where iteration
  # order is nondeterministic).
  if [[ -e "${m%.m}.sorted" ]]; then
    if diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch (sorted)"
      diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) | sed 's/^/  /'
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
  rm -f "$tmpll" "$tmpbin"
done

echo "----"
echo "run passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
