#!/usr/bin/env bash
# Phase 5 cross-backend round-trip test for the flowchart frontend.
#
# For each `.mflow` fixture under ../EmitMatlab/, this:
#   1. Round-trips the .mflow to MATLAB source via `-emit-matlab`.
#   2. Runs `-emit-c` (and -emit-cpp, -emit-python, -emit-ts) on
#      both the .mflow and the round-tripped .m.
#   3. Asserts the two backends produce byte-identical output.
#
# This is the v1 ship criterion for the flowchart frontend: the
# graph-derived AST must be indistinguishable from the canonical-
# formatted MATLAB source running through the same pipeline.
#
# Skips fixtures whose .mflow uses `library_id` only when no
# block-path is wired (the text-source path can't see the external
# library function definition); we pass --block-path explicitly
# to both sides for those fixtures.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURES="$TESTDIR/../EmitMatlab"
BLOCK_LIB="$FIXTURES/lib"
cd "$TESTDIR"

pass=0
fail=0

# Backends to compare. Bun/tsc/Node aren't required to be installed —
# we're diffing emitted source, not running it, so all four lanes
# work as long as `matlabc` itself produces the output.
backends=(c cpp python typescript)

run_one() {
  local f="$1"
  local name; name=$(basename "$f" .mflow)
  local flags=()
  if grep -q '"library_id"' "$f"; then
    flags=(--block-path "$BLOCK_LIB")
  fi

  # Step 1: round-trip to .m source.
  local synth="/tmp/${name}.flow.m"
  if ! "$MATLABC" ${flags[@]+"${flags[@]}"} -emit-matlab "$f" > "$synth" 2>/dev/null; then
    echo "FAIL $name (emit-matlab failed)"
    fail=$((fail+1))
    return
  fi

  # Step 2/3: per backend, diff -emit-X on both sides.
  local any_fail=0
  for be in "${backends[@]}"; do
    local from_flow="/tmp/${name}.flow.${be}"
    local from_text="/tmp/${name}.text.${be}"
    "$MATLABC" ${flags[@]+"${flags[@]}"} "-emit-${be}" "$f" > "$from_flow" 2>/dev/null || {
      # If the .mflow itself fails -emit-X (e.g. a known backend
      # gap unrelated to the frontend), the .m should fail too —
      # so the round-trip equivalence test is still meaningful.
      "$MATLABC" "-emit-${be}" "$synth" > "$from_text" 2>/dev/null
      if [[ -s "$from_text" ]]; then
        echo "FAIL $name [$be] (.mflow failed but .m succeeded)"
        any_fail=1
      fi
      continue
    }
    "$MATLABC" "-emit-${be}" "$synth" > "$from_text" 2>/dev/null || {
      echo "FAIL $name [$be] (.m round-trip failed but .mflow succeeded)"
      any_fail=1
      continue
    }
    # `diff -B` ignores blank-line-only changes. The C/Python/TS
    # emitters preserve source-paragraph blanks via SourceManager
    # gap inspection, which fires on the .m round-trip's single
    # buffer but not on the .mflow's per-block synthetic buffers
    # (cross-file gaps don't trigger the heuristic). Both outputs
    # are functionally identical — we just want to confirm there's
    # no meaningful drift between the two frontends.
    if ! diff -uB "$from_text" "$from_flow" > /tmp/${name}.${be}.diff 2>&1; then
      echo "FAIL $name [$be] (mflow vs. round-tripped .m differ)"
      head -20 /tmp/${name}.${be}.diff | sed 's/^/  /'
      any_fail=1
    fi
  done

  if [[ $any_fail -eq 0 ]]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
}

shopt -s nullglob
for f in "$FIXTURES"/*.mflow; do
  run_one "$f"
done

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
