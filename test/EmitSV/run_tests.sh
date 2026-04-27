#!/usr/bin/env bash
# Golden tests for the -emit-systemverilog backend (ASIC, Phase 1).
#
# For every .m in this directory, runs `matlabc -emit-systemverilog`
# and diffs stdout against the matching .sv.expected golden. When
# Verilator is on PATH (or VERILATOR is set), each emitted .sv is
# additionally lint-checked with `verilator --lint-only -Wall` —
# catches inferred-latch hazards and missing-width literals that the
# golden-diff lane alone cannot see.
#
# Usage: run_tests.sh <path-to-matlabc>
# Env:
#   UPDATE=1     regenerate every .sv.expected
#   VERILATOR=…  override or skip the lint pass (set empty to skip)
#                — defaults to the `verilator` on PATH if any
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

UPDATE="${UPDATE:-0}"
VERILATOR="${VERILATOR-$(command -v verilator || true)}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0; miss=0; lint_skip=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.sv.expected"
  # Verilator's DECLFILENAME warning fires whenever the on-disk
  # filename doesn't match the top module name; using `mktemp` would
  # produce `mlc.XXXXXX.sv` and trip every lint. Drop into a per-test
  # tempdir whose file is `<base>.sv`.
  tmpdir="$(mktemp -d -t emitsv.XXXXXX)"
  tmp="$tmpdir/$base.sv"

  if ! "$MATLABC" -emit-systemverilog "$m" > "$tmp" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-systemverilog exited non-zero"
    fail=$((fail+1))
    rm -rf "$tmpdir"
    continue
  fi

  if [[ "$UPDATE" == "1" ]]; then
    mv "$tmp" "$exp"
    echo "UPDATE $base"
    pass=$((pass+1))
    continue
  fi

  if [[ ! -e "$exp" ]]; then
    echo "MISS $base: no golden (rerun with UPDATE=1)"
    miss=$((miss+1))
    rm -rf "$tmpdir"
    continue
  fi

  if ! diff -u "$exp" "$tmp" >/dev/null; then
    echo "FAIL $base: golden drift"
    diff -u "$exp" "$tmp" | sed 's/^/  /'
    fail=$((fail+1))
    rm -rf "$tmpdir"
    continue
  fi

  # Verilator lint smoke. Top module name = file basename so the
  # DECLFILENAME warning is silenced when they match.
  if [[ -n "$VERILATOR" && -x "$VERILATOR" ]]; then
    if ! "$VERILATOR" --lint-only -Wall --top-module "$base" "$tmp" >/dev/null 2>&1; then
      echo "FAIL $base: verilator lint"
      "$VERILATOR" --lint-only -Wall --top-module "$base" "$tmp" 2>&1 | sed 's/^/  /'
      fail=$((fail+1))
      rm -rf "$tmpdir"
      continue
    fi
  else
    lint_skip=$((lint_skip+1))
  fi

  pass=$((pass+1))
  rm -f "$tmp"
done

echo "----"
if [[ $lint_skip -gt 0 ]]; then
  echo "emit-sv passed: $pass    failed: $fail    missing: $miss    (verilator skipped on $lint_skip)"
else
  echo "emit-sv passed: $pass    failed: $fail    missing: $miss    (verilator lint included)"
fi
exit $(( fail > 0 || miss > 0 ? 1 : 0 ))
