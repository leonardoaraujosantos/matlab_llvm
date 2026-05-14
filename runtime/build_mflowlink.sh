#!/usr/bin/env bash
# Build a Tier-G `-emit-mflowlink-cpp`-generated simulator.
#
# Usage:
#   runtime/build_mflowlink.sh <generated.cpp> [<output-binary>]
#
# Compiles the generated C++ against the matlab_llvm Flowchart static
# libs. The resulting binary is fully self-contained: it embeds the
# .mflow JSON at build time, so it can be deployed without the
# original source file or the matlab_llvm tree.
#
# Picks `clang++` if it's on the PATH, falls back to `c++`. Set
# CXX=<compiler> to override.

set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <generated.cpp> [<output-binary>]" >&2
  exit 2
fi

SRC="$1"
OUT="${2:-${SRC%.cpp}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INC="$ROOT/include"
LIB="$ROOT/build"

if [[ ! -f "$LIB/libMatlabFlowchart.a" || ! -f "$LIB/libMatlabBasic.a" ]]; then
  echo "error: matlab_llvm build artefacts not found under $LIB" >&2
  echo "       run \`ninja -C $LIB\` first (or pass MATLAB_LLVM_BUILD=...)" >&2
  exit 1
fi

CXX="${CXX:-$(command -v clang++ || command -v c++)}"
if [[ -z "$CXX" ]]; then
  echo "error: no C++ compiler found on PATH (clang++ / c++)" >&2
  exit 1
fi

"$CXX" -std=c++17 -O2 -I "$INC" "$SRC" \
       "$LIB/libMatlabFlowchart.a" "$LIB/libMatlabBasic.a" \
       -o "$OUT"

echo "built: $OUT"
