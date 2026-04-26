#!/usr/bin/env bash
# Build-and-run tests for the TypeScript emission path. For each .m in
# this directory, runs `matlabc -emit-typescript` and executes the
# emitted source with `tsx` (preferred) or `bun`, falling back to
# `ts-node` if neither is on PATH. Compares stdout against the matching
# .stdout file (or .stdout-typescript override). Programs that are
# known to diverge (e.g. MATLAB-bit-exact numerics) can be marked with
# a .skip-emit-typescript companion file and are reported as SKIPs.
#
# Per-test TypeScript-specific golden: matrix display through the TS
# runtime mirrors `%g` formatting; we still allow `<name>.stdout-typescript`
# overrides for cases where rounding differs from MATLAB.
#
# Set UPDATE=1 to (re)generate `.stdout-typescript` from the current
# emit whenever it diverges from `.stdout`. Useful after intentional
# emitter changes.
#
# Usage: run_tests_emitts.sh <path-to-matlabc>
# Env:   TSX=tsx          interpreter (default: tsx, then bun, then ts-node)
#        UPDATE=1         write .stdout-typescript on diff (off by default)
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

# Pick a TS interpreter. Each one knows how to handle .ts source files
# and basic ESM imports without configuration.
pick_runner() {
  if command -v tsx >/dev/null 2>&1; then echo "tsx"; return; fi
  if command -v bun >/dev/null 2>&1; then echo "bun run"; return; fi
  if command -v ts-node >/dev/null 2>&1; then echo "ts-node --transpile-only"; return; fi
  echo ""
}
RUNNER="${TSX:-$(pick_runner)}"
if [[ -z "$RUNNER" ]]; then
  echo "SKIP: no TypeScript runner found on PATH (tried tsx, bun, ts-node)" >&2
  exit 0
fi

UPDATE="${UPDATE:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

pass=0; fail=0; skip=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp_ts="${m%.m}.stdout-typescript"
  exp_default="${m%.m}.stdout"
  if [[ -e "$exp_ts" ]]; then
    exp="$exp_ts"
  elif [[ -e "$exp_default" ]]; then
    exp="$exp_default"
  else
    echo "SKIP $base (no .stdout)"; continue
  fi

  if [[ -e "${m%.m}.skip-emit-typescript" ]]; then
    echo "SKIP $base (marked .skip-emit-typescript)"
    skip=$((skip+1)); continue
  fi

  # Emit into the runtime dir so relative `./matlab_runtime` /
  # `./numpy_ts` imports resolve.
  tmpsrc="$(mktemp -t mlts.XXXXXX).ts"
  # Move the file into the runtime dir (where the imports point) so
  # ESM module resolution finds them; keep a symlink-style approach
  # via a temp file that imports from absolute path.
  tmpsrc_in_runtime="$RUNTIME_DIR/__test_$(basename "$tmpsrc")"

  if ! "$MATLABC" -emit-typescript "$m" > "$tmpsrc" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-typescript errored"
    fail=$((fail+1))
    rm -f "$tmpsrc"; continue
  fi
  cp "$tmpsrc" "$tmpsrc_in_runtime"

  got="$(cd "$RUNTIME_DIR" && $RUNNER "$tmpsrc_in_runtime" 2>/dev/null)" || {
    echo "FAIL $base: $RUNNER non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpsrc" "$tmpsrc_in_runtime"; continue
  }
  rm -f "$tmpsrc_in_runtime"

  if [[ -e "${m%.m}.sorted" ]]; then
    if diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      if [[ "$UPDATE" == "1" ]]; then
        printf '%s\n' "$got" > "$exp_ts"
        echo "UPDATE $base: wrote $(basename "$exp_ts")"
        pass=$((pass+1))
      else
        fail=$((fail+1))
        echo "FAIL $base: stdout mismatch (sorted)"
      fi
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    if [[ "$UPDATE" == "1" ]]; then
      printf '%s\n' "$got" > "$exp_ts"
      echo "UPDATE $base: wrote $(basename "$exp_ts")"
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch"
    fi
  fi
  rm -f "$tmpsrc"
done

echo "----"
echo "emit-typescript passed: $pass    failed: $fail    skipped: $skip"
exit $(( fail > 0 ? 1 : 0 ))
