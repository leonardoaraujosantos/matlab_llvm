#!/usr/bin/env bash
# Tier-C smoke lane — `matlabc -simulate model.mflow` runs the
# in-process simulation interpreter (lib/Flowchart/MflowLinkSim.cpp)
# and writes the logged signals as CSV. We do not golden-diff the
# CSV (floating-point output isn't stable across architectures);
# instead we sanity-check the header, the row count, and a couple
# of analytically-known values.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi
MATLABC="$(cd "$(dirname "$MATLABC")" && pwd)/$(basename "$MATLABC")"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/examples/mflowlink"

pass=0; fail=0
check() {
  if [[ "$2" == "$3" ]] || (eval "$2"); then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL  $1"
    echo "  expr: $2"
    [[ -n "${3:-}" ]] && echo "  expected: $3"
  fi
}

#--- lowpass: sine in → first-order LPF -----------------------------------
LP="$("$MATLABC" -simulate "$EX/lowpass.mflow")"
LP_HEAD=$(printf '%s\n' "$LP" | head -1)
LP_ROWS=$(printf '%s\n' "$LP" | wc -l | tr -d ' ')
check "lowpass header"     "[[ '$LP_HEAD' == 't,src,scope' ]]"  ""
check "lowpass row count"  "[[ $LP_ROWS -ge 1001 && $LP_ROWS -le 1003 ]]" ""
# Steady-state amplitude through 1/(s+1) for u = 2·sin(2π·t) is
# 2/sqrt(1+(2π)^2) ≈ 0.31427. Skip the t < 8 transient ramp; accept
# any peak |scope| over t ∈ [8, 10] in [0.30, 0.32].
LP_PEAK=$(printf '%s\n' "$LP" | awk -F, 'NR>1 && $1>=8 { v=$3<0?-$3:$3; if(v>m)m=v } END{print m+0}')
check "lowpass peak amp"   "awk 'BEGIN{exit !($LP_PEAK >= 0.30 && $LP_PEAK <= 0.32)}'" ""

#--- pid_tracking: step ref + PI controller → 1st-order plant -------------
PID="$("$MATLABC" -simulate "$EX/pid_tracking.mflow")"
PID_HEAD=$(printf '%s\n' "$PID" | head -1)
PID_LAST=$(printf '%s\n' "$PID" | tail -1 | awk -F, '{print $3}')
check "pid_tracking header" "[[ '$PID_HEAD' == 't,ref,out_scope' ]]" ""
check "pid tracks step"     "awk 'BEGIN{exit !($PID_LAST >= 0.98)}'" ""

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
