#!/usr/bin/env bash
# Tier-10 lint lane.  For each .m under examples/verilog_a/:
#   1. Compile via matlabc + clang and link against the matlab runtime.
#   2. Run the binary (which writes the .va file as a side effect).
#   3. Pipe the resulting .va files into scripts/va_lint.sh, which
#      delegates to OpenVAF (preferred) or ADMS (fallback).
#
# Skips cleanly (exit 0) if neither linter is installed.  Designed to
# be wired into CTest via the MATLAB_LLVM_WITH_VA_LINT CMake option.
#
# Usage: run_lint.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
CXX="${CXX:-${CLANG}++}"
EXAMPLES_DIR="$ROOT/examples/verilog_a"
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
  "$ROOT/runtime/toolbox/comm/runtime_comm.cpp"
  "$ROOT/runtime/toolbox/prop/runtime_prop.cpp"
  "$ROOT/runtime/toolbox/rf/runtime_rf.cpp"
)

if [[ ! -d "$EXAMPLES_DIR" ]]; then
  echo "skip: $EXAMPLES_DIR not found" >&2
  exit 0
fi

# Workdir for the emitted .va / .tbl files; we don't pollute
# examples/ so the gitignore policy stays clean.
WORKDIR="$(mktemp -d -t va_lint.XXXXXX)"
trap "rm -rf '$WORKDIR'" EXIT

fail=0
emitted=0
for m in "$EXAMPLES_DIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"

  tmpll="$WORKDIR/${base}.ll"
  tmpbin="$WORKDIR/${base}.out"
  if ! "$MATLABC" -emit-llvm "$m" > "$tmpll" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-llvm" >&2
    fail=$((fail+1)); continue
  fi
  if ! "$CXX" -Wno-override-module "$tmpll" "${RUNTIME_SRCS[@]}" \
        -I"$ROOT/runtime" -o "$tmpbin" 2>/dev/null; then
    echo "FAIL $base: clang link" >&2
    fail=$((fail+1)); continue
  fi
  # Run from WORKDIR so any sidecar .va / .tbl files land there.
  (cd "$WORKDIR" && "$tmpbin" >/dev/null 2>&1) || {
    echo "FAIL $base: binary exit nonzero" >&2
    fail=$((fail+1)); continue
  }
  emitted=$((emitted+1))
done

va_files=( "$WORKDIR"/*.va )
if (( ${#va_files[@]} == 0 )) || [[ ! -e "${va_files[0]}" ]]; then
  echo "no .va files emitted from $emitted example(s)" >&2
  exit 1
fi

"$ROOT/scripts/va_lint.sh" "${va_files[@]}"
lint_rc=$?

if (( fail > 0 )); then
  echo "$fail compile/run failure(s)" >&2
  exit 1
fi
exit $lint_rc
