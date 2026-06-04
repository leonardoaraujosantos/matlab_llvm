#!/usr/bin/env bash
# example_coverage.sh — measure how much of the examples/ corpus the compiler
# accepts through the AOT (LLVM) front end, and rank the gaps that block the
# rest. This turns the ad-hoc "% of examples that compile" sweep into a
# repeatable, tracked metric for production-readiness.
#
# It compiles each examples/**/*.m with `matlabc -emit-llvm` and classifies:
#   - OK            : front end + lowering produced LLVM with no error
#   - FAIL          : at least one diagnostic (undefined name / unsupported
#                     call shape / unconverted op)
#
# HDL and GPU examples are reported separately: they target SystemVerilog /
# GPU back ends (e.g. `global` hardware registers), so an AOT-only sweep is
# the wrong lane for them and they should not count against AOT coverage.
#
# Usage:
#   scripts/example_coverage.sh [path-to-matlabc]
#     (default matlabc: ./build/matlabc)
#
# Exit status is always 0 — this is a measurement tool, not a gate.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATLABC="${1:-$ROOT/build/matlabc}"
if [[ ! -x "$MATLABC" ]]; then
  echo "error: matlabc not found / not executable: $MATLABC" >&2
  echo "usage: $0 [path-to-matlabc]" >&2
  exit 2
fi

cd "$ROOT"

# Classify a path as a non-AOT target (HDL hardware / GPU back ends).
is_wrong_target() {
  case "$1" in
    */hdl/*|*/gpu/*|*gpu_*|*_gpu*) return 0 ;;
    *) return 1 ;;
  esac
}

gp_ok=0; gp_fail=0; ht_ok=0; ht_fail=0
undef_tally="$(mktemp)"; unsup_tally="$(mktemp)"; gp_fail_list="$(mktemp)"
trap 'rm -f "$undef_tally" "$unsup_tally" "$gp_fail_list"' EXIT

for f in $(find examples -name '*.m' | sort); do
  err="$("$MATLABC" -emit-llvm "$f" 2>&1 >/dev/null)"
  if is_wrong_target "$f"; then
    if [[ -z "$err" ]]; then ht_ok=$((ht_ok+1)); else ht_fail=$((ht_fail+1)); fi
    continue
  fi
  if [[ -z "$err" ]]; then
    gp_ok=$((gp_ok+1))
  else
    gp_fail=$((gp_fail+1))
    tok="$(printf '%s' "$err" | grep -oiE "undefined name '[^']*'|unsupported call shape for built-in function '[^']*'" | head -1)"
    echo "$f :: $tok" >> "$gp_fail_list"
    printf '%s\n' "$err" | grep -oiE "undefined name '[^']*'"                            >> "$undef_tally"
    printf '%s\n' "$err" | grep -oiE "unsupported call shape for built-in function '[^']*'" >> "$unsup_tally"
  fi
done

gp_total=$((gp_ok+gp_fail))
pct="n/a"
if [[ $gp_total -gt 0 ]]; then
  pct="$(awk "BEGIN{printf \"%.1f\", 100*$gp_ok/$gp_total}")"
fi

echo "=== examples AOT coverage (matlabc -emit-llvm) ==="
echo "General-purpose : $gp_ok / $gp_total OK  (${pct}%)"
echo "HDL/GPU (separate target, informational): $ht_ok OK / $ht_fail fail"
echo
echo "=== general-purpose failures ==="
if [[ -s "$gp_fail_list" ]]; then cat "$gp_fail_list"; else echo "(none)"; fi
echo
echo "=== top missing names (undefined) ==="
sort "$undef_tally" | uniq -c | sort -rn | head -15
echo
echo "=== top unsupported-call builtins ==="
sed -E "s/.*'([^']*)'.*/\1/" "$unsup_tally" | sort | uniq -c | sort -rn | head -15

exit 0
