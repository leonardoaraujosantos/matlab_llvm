#!/usr/bin/env bash
# Build matlabc with Clang source-based coverage, run the test suites,
# and produce a per-function coverage report scoped to the runtime
# translation units.
#
# Phase-2/2.5 file split (docs/port_runtime_2_cpp.md): the runtime
# now lives in three .cpp files (matlab_runtime.cpp, runtime_debug.cpp,
# runtime_complex.cpp). The script reports against all three.
#
# Usage:
#   scripts/runtime_coverage.sh           # build + run all + report
#   scripts/runtime_coverage.sh --no-run  # use already-recorded .profraw
#   scripts/runtime_coverage.sh --html    # also emit HTML report
#
# Output:
#   build-coverage/coverage/runtime.profdata  (merged profile)
#   build-coverage/coverage/summary.txt       (per-function table)
#   build-coverage/coverage/uncovered.txt     (functions with 0% lines)
#   build-coverage/coverage/html/index.html   (when --html)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-coverage"
COV="$BUILD/coverage"
PROFRAW_DIR="$COV/profraw"
RUNTIME_SRCS=(
  "$ROOT/runtime/matlab_runtime.cpp"
  "$ROOT/runtime/runtime_debug.cpp"
  "$ROOT/runtime/runtime_complex.cpp"
)

WANT_HTML=0
SKIP_RUN=0
for arg in "$@"; do
  case "$arg" in
    --html) WANT_HTML=1 ;;
    --no-run) SKIP_RUN=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Prefer Homebrew's llvm tools; Apple clang's profile lib is compatible
# with Homebrew llvm-profdata / llvm-cov.
LLVM_BIN_CANDIDATES=(
  "/opt/homebrew/opt/llvm/bin"
  "/usr/local/opt/llvm/bin"
  "$(dirname "$(command -v llvm-profdata 2>/dev/null || true)")"
)
LLVM_PROFDATA=""
LLVM_COV=""
for d in "${LLVM_BIN_CANDIDATES[@]}"; do
  [ -z "$d" ] && continue
  if [ -x "$d/llvm-profdata" ] && [ -x "$d/llvm-cov" ]; then
    LLVM_PROFDATA="$d/llvm-profdata"; LLVM_COV="$d/llvm-cov"; break
  fi
done
if [ -z "$LLVM_PROFDATA" ]; then
  echo "error: llvm-profdata / llvm-cov not found." >&2
  echo "install with: brew install llvm" >&2
  exit 1
fi
echo "==> using $LLVM_PROFDATA"
echo "==> using $LLVM_COV"

mkdir -p "$BUILD" "$COV" "$PROFRAW_DIR"

# Configure + build.
echo "==> configuring with -DMATLAB_LLVM_COVERAGE=ON"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMATLAB_LLVM_COVERAGE=ON >/dev/null

# Test binaries from the foreach() block in CMakeLists.txt — keep this
# list in sync with the IN ITEMS list there.
RT_TESTS=(linalg shape rng complex reduce signal stats image more fft
          struct_cell fi elementwise unary fi_arrays strings)

echo "==> building matlabc + runtime-test-*"
TARGETS=(matlabc)
for t in "${RT_TESTS[@]}"; do TARGETS+=("runtime-test-$t"); done
cmake --build "$BUILD" --target "${TARGETS[@]}" -j

MATLABC="$BUILD/matlabc"
[ -x "$MATLABC" ] || { echo "error: matlabc not built at $MATLABC" >&2; exit 1; }

# Each instrumented binary has its own signature in the profile data.
# Pass all of them to llvm-cov so the merged report covers every TU
# that executed runtime code.
COV_OBJECTS=( "$MATLABC" )
for t in "${RT_TESTS[@]}"; do
  if [ -x "$BUILD/runtime-test-$t" ]; then
    COV_OBJECTS+=( -object "$BUILD/runtime-test-$t" )
  fi
done

if [ "$SKIP_RUN" -eq 0 ]; then
  echo "==> wiping old .profraw"
  rm -f "$PROFRAW_DIR"/*.profraw

  # %p = pid, %m = binary signature; multi-process suites overlap cleanly.
  export LLVM_PROFILE_FILE="$PROFRAW_DIR/runtime-%p-%m.profraw"

  echo "==> running CTest"
  # Run as many suites as possible; we don't fail the report if a suite
  # fails — coverage is still useful. Use -V on failure for diagnosis.
  ctest --test-dir "$BUILD" --output-on-failure || \
    echo "==> warning: some tests failed; continuing with whatever ran"
fi

# Merge profraws.
shopt -s nullglob
PROFRAWS=( "$PROFRAW_DIR"/*.profraw )
shopt -u nullglob
if [ "${#PROFRAWS[@]}" -eq 0 ]; then
  echo "error: no .profraw files in $PROFRAW_DIR — did the suites run?" >&2
  exit 1
fi
echo "==> merging ${#PROFRAWS[@]} profraw files"
"$LLVM_PROFDATA" merge -sparse -o "$COV/runtime.profdata" "${PROFRAWS[@]}"

# Per-function summary, scoped to the runtime TU.
echo "==> generating per-function summary"
"$LLVM_COV" report "${COV_OBJECTS[@]}" \
  -instr-profile="$COV/runtime.profdata" \
  -sources "${RUNTIME_SRCS[@]}" \
  > "$COV/summary.txt"

# 0%-covered functions list. llvm-cov report -show-functions prints
# one row per function with columns:
#   Name  Regions Miss Cover  Lines Miss Cover  Branches Miss Cover
# A function is "0% line coverage" when the 7th column ("Lines Cover")
# is 0.00%. Skip the table header / divider rows.
"$LLVM_COV" report "${COV_OBJECTS[@]}" \
  -instr-profile="$COV/runtime.profdata" \
  -show-functions \
  -sources "${RUNTIME_SRCS[@]}" \
  | awk 'NF >= 10 && $1 ~ /^[A-Za-z_]/ && $7 == "0.00%" { print $1 }' \
  | sort -u > "$COV/uncovered.txt" || true

if [ "$WANT_HTML" -eq 1 ]; then
  echo "==> generating HTML report"
  "$LLVM_COV" show "${COV_OBJECTS[@]}" \
    -instr-profile="$COV/runtime.profdata" \
    -format=html \
    -output-dir="$COV/html" \
    -sources "${RUNTIME_SRCS[@]}"
fi

echo
echo "==> summary.txt:"
cat "$COV/summary.txt"
echo
echo "==> uncovered functions: $(wc -l < "$COV/uncovered.txt") (see $COV/uncovered.txt)"
[ "$WANT_HTML" -eq 1 ] && echo "==> html: $COV/html/index.html"
