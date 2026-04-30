#!/usr/bin/env bash
# Compile a .m source into an executable using matlabc + clang + the
# matlab runtime shim, then run it.
#
# Usage: build_and_run.sh <input.m> [output-name]
#
# Environment:
#   MATLABC   path to the matlabc binary (default: build/matlabc)
#   CLANG     path to clang             (default: clang in PATH)
set -euo pipefail

MATLABC="${MATLABC:-$(cd "$(dirname "$0")/.." && pwd)/build/matlabc}"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
RUNTIME="$(cd "$(dirname "$0")" && pwd)/matlab_runtime.cpp"

if [[ ! -x "$MATLABC" ]]; then
  echo "error: matlabc not found at $MATLABC" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <input.m> [output]" >&2
  exit 64
fi

INPUT="$1"
OUT="${2:-$(basename "${INPUT%.m}")}"
TMP="$(mktemp -t matlabc.XXXXXX).ll"
trap 'rm -f "$TMP"' EXIT

# Runtime is C++ since Phase 3 of docs/port_runtime_2_cpp.md. Drive
# the link line with clang++ so the .cpp gets compiled as C++; the
# matlabc-emitted .ll is still C-compatible.
"$MATLABC" -emit-llvm "$INPUT" > "$TMP"
"${CLANG}++" -Wno-override-module "$TMP" "$RUNTIME" -o "$OUT"
echo "built $OUT"
