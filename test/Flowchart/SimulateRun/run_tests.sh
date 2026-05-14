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

#--- multirate (Tier E): sine -> ZOH @ 0.1 s -> 1st-order plant -----------
MR="$("$MATLABC" -simulate "$EX/multirate.mflow")"
# The ZOH samples sin(2π·t) at t=0.1, expecting sin(0.628)≈0.5878.
# Accept any value in [0.50, 0.65] across the t=0.1 sample window.
MR_ZOH_01=$(printf '%s\n' "$MR" | awk -F, 'NR>1 && $1+0 >= 0.10 && $1+0 < 0.11 { print $3; exit }')
check "ZOH @ 0.1s = sin(0.628)" \
  "awk 'BEGIN{exit !($MR_ZOH_01 >= 0.50 && $MR_ZOH_01 <= 0.65)}'" ""
# The ZOH must HOLD between sample ticks — same value at 0.10 and 0.15.
MR_ZOH_015=$(printf '%s\n' "$MR" | awk -F, 'NR>1 && $1+0 >= 0.145 && $1+0 < 0.155 { print $3; exit }')
check "ZOH holds between ticks" \
  "awk 'BEGIN{exit !(($MR_ZOH_01 - $MR_ZOH_015)^2 < 1e-9)}'" ""

#--- freefall_floor (Tier E): integrator-driven zero-crossing on saturation
FF="$("$MATLABC" -simulate "$EX/freefall_floor.mflow")"
FF_HEAD=$(printf '%s\n' "$FF" | head -1)
check "freefall header"        "[[ '$FF_HEAD' == 't,pos,floor,scope' ]]" ""
# pos starts at 10, floor saturation identity at t=0 = 10.
FF_T0=$(printf '%s\n' "$FF" | awk -F, 'NR==2 { print $3 }')
check "freefall floor@t=0"     "awk 'BEGIN{exit !($FF_T0 >= 9.99 && $FF_T0 <= 10.01)}'" ""
# By t≈1.5 the ball has hit the floor; saturation clamps `floor` to 0.
FF_LATE=$(printf '%s\n' "$FF" | awk -F, '$1+0 >= 1.5 && $1+0 < 1.51 { print $3; exit }')
check "freefall floor clamps"  "awk 'BEGIN{exit !($FF_LATE*$FF_LATE < 1e-3)}'" ""

#--- enabled_subsystem (Tier F): gate flips at t=2 ---------------------
EN="$("$MATLABC" -simulate "$EX/enabled_subsystem.mflow")"
EN_HEAD=$(printf '%s\n' "$EN" | head -1)
check "enabled header"   "[[ '$EN_HEAD' == 't,drive,gate,sub/amp,scope' ]]" ""
# Before t=2 the scope is held at zero by the gate.
EN_PRE=$(printf '%s\n' "$EN" | awk -F, '$1+0 >= 0.99 && $1+0 < 1.01 { print $5; exit }')
check "scope held pre-gate" "awk 'BEGIN{exit !($EN_PRE*$EN_PRE < 1e-9)}'" ""
# After t=2.01 the sub fires: scope = 2*drive (gain=2).
EN_POST=$(printf '%s\n' "$EN" | awk -F, '$1+0 >= 2.10 && $1+0 < 2.12 { print $5; prev=$2; exit } { prev=$2 }')
EN_DRIVE=$(printf '%s\n' "$EN" | awk -F, '$1+0 >= 2.10 && $1+0 < 2.12 { print $2; exit }')
check "scope = 2*drive post-gate" \
  "awk 'BEGIN{d=$EN_DRIVE; s=$EN_POST; exit !((s - 2*d)^2 < 1e-6)}'" ""

#--- saturation_zc (Tier E): ramp -1.5..2.5 through ±1 saturation ----------
SZ="$("$MATLABC" -simulate "$EX/saturation_zc.mflow")"
# At t = 1.5 the ramp = 0 (well within the rails), sat output = 0.
SZ_MID=$(printf '%s\n' "$SZ" | awk -F, '$1+0 >= 1.495 && $1+0 < 1.505 { print $3; exit }')
check "sat passthrough mid"  "awk 'BEGIN{exit !($SZ_MID*$SZ_MID < 1e-3)}'" ""
# At t = 3.5 the ramp = 2.0, sat clamps to +1.0.
SZ_HI=$(printf '%s\n' "$SZ" | awk -F, '$1+0 >= 3.495 && $1+0 < 3.505 { print $3; exit }')
check "sat clamps to upper"  "awk 'BEGIN{exit !($SZ_HI >= 0.999 && $SZ_HI <= 1.001)}'" ""

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
