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
check "lowpass header"     "[[ '$LP_HEAD' == 't,src,scope'* ]]"  ""
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

#--- triggered_counter (Tier F carve-out): rising-edge samples a ramp ------
TR="$("$MATLABC" -simulate "$EX/triggered_counter.mflow")"
TR_HEAD=$(printf '%s\n' "$TR" | head -1)
check "triggered header" "[[ '$TR_HEAD' == 't,clk,src,trig/pass,scope' ]]" ""
# The scope should latch the ramp value at integer t (= 1, 2, 3, 4, 5).
# Outside those instants, the gate is closed and the scope reads 0.
TR_T1=$(printf '%s\n' "$TR" | awk -F, '$1+0==1 { print $5; exit }')
TR_T3=$(printf '%s\n' "$TR" | awk -F, '$1+0==3 { print $5; exit }')
TR_T5=$(printf '%s\n' "$TR" | awk -F, '$1+0==5 { print $5; exit }')
check "trigger@1 latches ramp(1)" \
  "awk 'BEGIN{exit !(($TR_T1 - 1)^2 < 1e-6)}'" ""
check "trigger@3 latches ramp(3)" \
  "awk 'BEGIN{exit !(($TR_T3 - 3)^2 < 1e-6)}'" ""
check "trigger@5 latches ramp(5)" \
  "awk 'BEGIN{exit !(($TR_T5 - 5)^2 < 1e-6)}'" ""
# And between triggers (say t = 1.5), the scope is held at 0 (the
# edge-triggered subsystem resets its output outside the firing
# step — Simulink's "Output when disabled: reset" mode).
TR_BTW=$(printf '%s\n' "$TR" | awk -F, '$1+0 >= 1.49 && $1+0 < 1.51 { print $5; exit }')
check "trigger off between" \
  "awk 'BEGIN{exit !($TR_BTW*$TR_BTW < 1e-12)}'" ""

#--- thermostat (Tier E carve-out): hysteretic relay bang-bang -------------
TH="$("$MATLABC" -simulate "$EX/thermostat.mflow")"
TH_HEAD=$(printf '%s\n' "$TH" | head -1)
check "thermostat header" "[[ '$TH_HEAD' == 't,ctrl,plant,scope' ]]" ""
# The plant must oscillate between [19.5, 20.5] in steady state.
TH_MAX=$(printf '%s\n' "$TH" | awk -F, 'NR>1 && $1+0>=3 {if($3>m)m=$3} END{print m+0}')
TH_MIN=$(printf '%s\n' "$TH" | awk -F, 'NR>1 && $1+0>=3 {if(m==""||$3<m)m=$3} END{print m+0}')
check "thermostat upper bound" \
  "awk 'BEGIN{exit !($TH_MAX >= 20.49 && $TH_MAX <= 20.52)}'" ""
check "thermostat lower bound" \
  "awk 'BEGIN{exit !($TH_MIN >= 19.48 && $TH_MIN <= 19.51)}'" ""
# The relay control output must visit BOTH 0 and 20 in steady state
# (i.e. it actually bangs back and forth, not stuck on one rail).
TH_ON=$(printf '%s\n' "$TH" | awk -F, 'NR>1 && $1+0>=3 && $2+0>=19' | head -1)
TH_OFF=$(printf '%s\n' "$TH" | awk -F, 'NR>1 && $1+0>=3 && $2+0<=1' | head -1)
check "relay reaches on rail"  "[[ -n '$TH_ON' ]]"  ""
check "relay reaches off rail" "[[ -n '$TH_OFF' ]]" ""

#--- saturation_zc (Tier E): ramp -1.5..2.5 through ±1 saturation ----------
SZ="$("$MATLABC" -simulate "$EX/saturation_zc.mflow")"
# At t = 1.5 the ramp = 0 (well within the rails), sat output = 0.
SZ_MID=$(printf '%s\n' "$SZ" | awk -F, '$1+0 >= 1.495 && $1+0 < 1.505 { print $3; exit }')
check "sat passthrough mid"  "awk 'BEGIN{exit !($SZ_MID*$SZ_MID < 1e-3)}'" ""
# At t = 3.5 the ramp = 2.0, sat clamps to +1.0.
SZ_HI=$(printf '%s\n' "$SZ" | awk -F, '$1+0 >= 3.495 && $1+0 < 3.505 { print $3; exit }')
check "sat clamps to upper"  "awk 'BEGIN{exit !($SZ_HI >= 0.999 && $SZ_HI <= 1.001)}'" ""

#--- tier_h_showcase: clock / trig / math / compare / lookup / delay --------
HS="$("$MATLABC" -simulate "$EX/tier_h_showcase.mflow")"
HS_HEAD=$(printf '%s\n' "$HS" | head -1)
check "tier-H showcase header" \
  "[[ '$HS_HEAD' == 't,clk,sin_t,abs_sin,gt_half,lut_table,delay,scope' ]]" ""
# At t=1: clk=1, sin(1)=0.841, |sin|=0.841 > 0.5 ⇒ 1, lookup(1)=1.
HS_AT1=$(printf '%s\n' "$HS" | awk -F, '$1+0==1 {print; exit}')
HS_SIN=$(printf '%s\n' "$HS_AT1" | awk -F, '{print $3}')
HS_ABS=$(printf '%s\n' "$HS_AT1" | awk -F, '{print $4}')
HS_GT=$(printf '%s\n'  "$HS_AT1" | awk -F, '{print $5}')
HS_LUT=$(printf '%s\n' "$HS_AT1" | awk -F, '{print $6}')
check "trig sin(1) == 0.8414" \
  "awk 'BEGIN{exit !(($HS_SIN - 0.8414709)^2 < 1e-6)}'" ""
check "math abs of sin >= 0" \
  "awk 'BEGIN{exit !($HS_ABS >= 0)}'" ""
check "compare >0.5 fires at sin(1)" "[[ '$HS_GT' == '1.000000000e+00' ]]" ""
check "lookup_1d(1) interpolates 1" \
  "awk 'BEGIN{exit !(($HS_LUT - 1)^2 < 1e-9)}'" ""
# transport_delay(0.25 s): at t = 1.0 the scope reads lookup at t = 0.75
# = linear interp between (0,0) and (1,1) = 0.75.
HS_DEL=$(printf '%s\n' "$HS" | awk -F, '$1+0==1 {print $7; exit}')
check "transport_delay shifts 0.25s" \
  "awk 'BEGIN{exit !(($HS_DEL - 0.75)^2 < 1e-6)}'" ""
# lookup_1d at x=3 should hit the table entry exactly = 9.
HS_LUT3=$(printf '%s\n' "$HS" | awk -F, '$1+0==3 {print $6; exit}')
check "lookup_1d(3) == 9" \
  "awk 'BEGIN{exit !(($HS_LUT3 - 9)^2 < 1e-9)}'" ""

#--- tier_h_logic: AND/OR/XOR/<  truth table ---------------------------------
LG="$("$MATLABC" -simulate "$EX/tier_h_logic.mflow")"
LG_HEAD=$(printf '%s\n' "$LG" | head -1)
check "tier-H logic header" \
  "[[ '$LG_HEAD' == 't,and_ab,or_ab,xor_ab,rel_lt' ]]" ""
# t=0.5: A=0, B=0 → AND=0, OR=0, XOR=0, rel_lt(0<0)=0
LG_R05=$(printf '%s\n' "$LG" | awk -F, '$1+0==0.5 {print $2","$3","$4","$5; exit}')
check "logic A=0,B=0 row" "[[ '$LG_R05' == '0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00' ]]" ""
# t=1.5: A=1, B=0 → AND=0, OR=1, XOR=1, rel_lt(1<1)=0
LG_R15=$(printf '%s\n' "$LG" | awk -F, '$1+0==1.5 {print $2","$3","$4","$5; exit}')
check "logic A=1,B=0 row" "[[ '$LG_R15' == '0.000000000e+00,1.000000000e+00,1.000000000e+00,0.000000000e+00' ]]" ""
# t=2.5: A=1, B=1 → AND=1, OR=1, XOR=0, rel_lt(0<1)=1
LG_R25=$(printf '%s\n' "$LG" | awk -F, '$1+0==2.5 {print $2","$3","$4","$5; exit}')
check "logic A=1,B=1 row" "[[ '$LG_R25' == '1.000000000e+00,1.000000000e+00,0.000000000e+00,1.000000000e+00' ]]" ""

#--- tier_h_discrete: ForwardEuler accumulator ------------------------------
DI="$("$MATLABC" -simulate "$EX/tier_h_discrete.mflow")"
DI_HEAD=$(printf '%s\n' "$DI" | head -1)
check "tier-H discrete header" "[[ '$DI_HEAD' == 't,di,scope' ]]" ""
# After 10 ticks of period 0.1 with input=1, di = 1.0.
DI_T1=$(printf '%s\n' "$DI" | awk -F, '$1+0==1 {print $2; exit}')
check "discrete_integrator hits 1.0 at t=1" \
  "awk 'BEGIN{exit !(($DI_T1 - 1)^2 < 1e-9)}'" ""
# After 20 ticks, di = 2.0.
DI_T2=$(printf '%s\n' "$DI" | awk -F, '$1+0==2 {print $2; exit}')
check "discrete_integrator hits 2.0 at t=2" \
  "awk 'BEGIN{exit !(($DI_T2 - 2)^2 < 1e-9)}'" ""

#--- goto_from: virtual-wire elision -------------------------------------
GF="$("$MATLABC" -simulate "$EX/goto_from.mflow")"
GF_HEAD=$(printf '%s\n' "$GF" | head -1)
check "goto/from header"  "[[ '$GF_HEAD' == 't,src,amp_a,amp_b,scope' ]]" ""
# At t = 0.5: src = sin(π·0.5) = 1, so amp_a = 2 and amp_b = -3,
# scope = 2 + (-3) = -1. Confirms the from-blocks were rewired to
# the goto's source and the goto+from pair is gone from the IR.
GF_T05=$(printf '%s\n' "$GF" | awk -F, '$1+0==0.5 { print $5; exit }')
check "goto/from scope at t=0.5 == -1" \
  "awk 'BEGIN{exit !(($GF_T05 + 1)^2 < 1e-9)}'" ""

#--- matlab_fcn: expression evaluator ------------------------------------
MF="$("$MATLABC" -simulate "$EX/matlab_fcn.mflow")"
MF_HEAD=$(printf '%s\n' "$MF" | head -1)
check "matlab_fcn header"  "[[ '$MF_HEAD' == 't,src1,src2,fcn,scope' ]]" ""
# At t = 0: u1=0, u2=0 → 0·sin(0) + sqrt(0) - 1 = -1.0
MF_T0=$(printf '%s\n' "$MF" | awk -F, 'NR>1 && $1+0==0 { print $4; exit }')
check "matlab_fcn(0,0) == -1"   "awk 'BEGIN{exit !(($MF_T0 + 1)^2 < 1e-9)}'" ""
# At t = 0.5: u1=1, u2=0.25 → 1·sin(π) + sqrt(0.25) - 1 = -0.5
MF_T05=$(printf '%s\n' "$MF" | awk -F, '$1+0==0.5 { print $4; exit }')
check "matlab_fcn at t=0.5 == -0.5" \
  "awk 'BEGIN{exit !(($MF_T05 + 0.5)^2 < 1e-6)}'" ""
# At t = 2: u1≈0, u2=1.0 → 0·sin(4π) + sqrt(1) - 1 = 0
MF_T2=$(printf '%s\n' "$MF" | awk -F, '$1+0==2 { print $4; exit }')
check "matlab_fcn at t=2 == 0"      "awk 'BEGIN{exit !($MF_T2^2 < 1e-9)}'" ""
# At t = 3: u1≈0, u2=1.5 → sqrt(1.5) - 1 ≈ 0.2247
MF_T3=$(printf '%s\n' "$MF" | awk -F, '$1+0==3 { print $4; exit }')
check "matlab_fcn at t=3 ≈ 0.2247" \
  "awk 'BEGIN{exit !(($MF_T3 - 0.2247448)^2 < 1e-6)}'" ""

#--- vector_signals (Item-1): width inference + Mux concat ----------------
VS="$("$MATLABC" -simulate "$EX/vector_signals.mflow")"
VS_HEAD=$(printf '%s\n' "$VS" | head -1)
# 1 (t) + 3 (src_const) + 3 (amp) + 4 (mux) + 4 (scope) = 15 columns.
VS_COLS=$(printf '%s' "$VS_HEAD" | awk -F, '{print NF}')
check "vector header has 15 columns" "[[ '$VS_COLS' == '15' ]]" ""
# At t=0: gain·[1,2,3] = [2,4,6]; mux = [2,4,6,sin(0)=0]; scope mirrors.
VS_AMP1=$(printf '%s\n' "$VS" | awk -F, 'NR==2 { print $5 }')
VS_AMP3=$(printf '%s\n' "$VS" | awk -F, 'NR==2 { print $7 }')
VS_MUX4_T0=$(printf '%s\n' "$VS" | awk -F, 'NR==2 { print $11 }')
check "amp[1]=2 at t=0" \
  "awk 'BEGIN{exit !(($VS_AMP1 - 2)^2 < 1e-9)}'" ""
check "amp[3]=6 at t=0" \
  "awk 'BEGIN{exit !(($VS_AMP3 - 6)^2 < 1e-9)}'" ""
check "mux[4]=0 at t=0 (sin(0))" \
  "awk 'BEGIN{exit !($VS_MUX4_T0^2 < 1e-9)}'" ""
# At t=0.5: mux[4] = sin(π·0.5) = 1.
VS_MUX4=$(printf '%s\n' "$VS" | awk -F, 'NR>1 && $1+0==0.5 { print $11; exit }')
check "mux[4] at t=0.5 ≈ 1" \
  "awk 'BEGIN{exit !(($VS_MUX4 - 1)^2 < 1e-9)}'" ""
VS_SC4=$(printf '%s\n' "$VS" | awk -F, 'NR>1 && $1+0==0.5 { print $15; exit }')
check "scope[4] mirrors mux[4]" \
  "awk 'BEGIN{exit !(($VS_SC4 - 1)^2 < 1e-9)}'" ""

#--- sample_time_inherit (Item-1): downstream gain inherits 0.1 from ZOH ---
ST_LINE=$("$MATLABC" -simulate --dry-run "$EX/sample_time_inherit.mflow" 2>&1 | grep 'gain kind=signal_gain')
check "gain block inherits discrete 0.1s" \
  "[[ '$ST_LINE' == *'sample=discrete period=0.1'* ]]" ""

#--- discrete_integrator_methods (§17.5 #4): three methods on a ramp ------
DM="$("$MATLABC" -simulate "$EX/discrete_integrator_methods.mflow")"
DM_HEAD=$(printf '%s\n' "$DM" | head -1)
check "di-methods header"   "[[ '$DM_HEAD' == 't,u_ramp,di_fe,di_be,di_tr' ]]" ""
# Integrate u(t)=t from 0 to 2 with h=0.1: Forward Euler = Σh·u[n]
# for n=0..19 = 1.9. Backward Euler = Σh·u[n+1] = 2.1. Trapezoidal
# is exact = 2.0.
DM_FE=$(printf '%s\n' "$DM" | awk -F, 'NR>1 && $1+0==2.0 {print $3; exit}')
DM_BE=$(printf '%s\n' "$DM" | awk -F, 'NR>1 && $1+0==2.0 {print $4; exit}')
DM_TR=$(printf '%s\n' "$DM" | awk -F, 'NR>1 && $1+0==2.0 {print $5; exit}')
check "Forward Euler (lagging)  = 1.9" \
  "awk 'BEGIN{exit !(($DM_FE - 1.9)^2 < 1e-6)}'" ""
check "Backward Euler (leading) = 2.1" \
  "awk 'BEGIN{exit !(($DM_BE - 2.1)^2 < 1e-6)}'" ""
check "Trapezoidal (exact)      = 2.0" \
  "awk 'BEGIN{exit !(($DM_TR - 2.0)^2 < 1e-9)}'" ""

#--- masked_library (Item-3): two subsystem instances of one library ------
ML="$("$MATLABC" -simulate "$EX/masked_library.mflow")"
ML_HEAD=$(printf '%s\n' "$ML" | head -1)
check "masked header"        "[[ '$ML_HEAD' == 't,src,fast_scope,slow_scope' ]]" ""
# fast LP: τ=0.1, ωc=10, amplitude 1/√(1+(2π/10)²) ≈ 0.847.
ML_FAST=$(printf '%s\n' "$ML" | awk -F, 'NR>1 && $1+0>=3 { v=$3<0?-$3:$3; if(v>m)m=v } END {print m}')
check "fast LP peak ≈ 0.847" \
  "awk 'BEGIN{exit !(($ML_FAST - 0.847)^2 < 1e-3)}'" ""
# slow LP: τ=1.0, ωc=1, amplitude 1/√(1+(2π)²) ≈ 0.157.
ML_SLOW=$(printf '%s\n' "$ML" | awk -F, 'NR>1 && $1+0>=3 { v=$4<0?-$4:$4; if(v>m)m=v } END {print m}')
check "slow LP peak ≈ 0.157" \
  "awk 'BEGIN{exit !(($ML_SLOW - 0.157)^2 < 1e-3)}'" ""

#--- matlab_function_block (Item-4): soft-saturated mixer with deadband ---
FN="$("$MATLABC" -simulate "$EX/matlab_function_block.mflow")"
FN_HEAD=$(printf '%s\n' "$FN" | head -1)
check "matlab_function header" \
  "[[ '$FN_HEAD' == 't,src_a,src_b,mixer,scope' ]]" ""
# At t=0: a=0, b=0, s=0 → |s|<0.2 → deadband branch → y=0.
FN_DB=$(printf '%s\n' "$FN" | awk -F, 'NR>1 && $1+0==0 {print $4; exit}')
check "deadband branch: y=0 at s=0" \
  "awk 'BEGIN{exit !($FN_DB^2 < 1e-9)}'" ""
# At t=0.5: a=sin(π·0.5)·1.2=1.2, b≈0, s=1.2 → elseif s>1 → y = 1 + 0.1·(1.2-1) = 1.02.
FN_POS=$(printf '%s\n' "$FN" | awk -F, 'NR>1 && $1+0==0.5 {print $4; exit}')
check "soft-sat positive: y=1+0.1·(s-1)" \
  "awk 'BEGIN{exit !(($FN_POS - 1.02)^2 < 1e-5)}'" ""
# At t=1.5: a≈-1.2 → s≈-1.2 → elseif s<-1 → y = -1 + 0.1·(-1.2+1) = -1.02.
FN_NEG=$(printf '%s\n' "$FN" | awk -F, 'NR>1 && $1+0==1.5 {print $4; exit}')
check "soft-sat negative: y=-1+0.1·(s+1)" \
  "awk 'BEGIN{exit !(($FN_NEG + 1.02)^2 < 1e-5)}'" ""

#--- discrete_pid (Item-1 follow-on): 50ms-sampled PID + cont plant -------
DP="$("$MATLABC" -simulate "$EX/discrete_pid.mflow")"
DP_HEAD=$(printf '%s\n' "$DP" | head -1)
check "discrete_pid header" \
  "[[ '$DP_HEAD' == 't,ref,sampler,i_acc,u_sampled,plant,scope' ]]" ""
# Sample-time inheritance: kp / ki / u_sampled / plant all picked up
# `discrete period=0.05` from the upstream ZOH.
DP_KP=$("$MATLABC" -simulate --dry-run "$EX/discrete_pid.mflow" 2>&1 | grep 'kp kind=signal_gain')
check "kp inherits discrete 0.05s" \
  "[[ '$DP_KP' == *'sample=discrete period=0.05'* ]]" ""
DP_USUM=$("$MATLABC" -simulate --dry-run "$EX/discrete_pid.mflow" 2>&1 | grep 'u_sampled kind=signal_sum')
check "u_sampled inherits discrete 0.05s" \
  "[[ '$DP_USUM' == *'sample=discrete period=0.05'* ]]" ""
# Steady-state: plant approaches reference (1.0) within tolerance.
DP_END=$(printf '%s\n' "$DP" | awk -F, 'END{print $6}')
check "discrete PID converges past 0.9" \
  "awk 'BEGIN{exit !($DP_END > 0.9 && $DP_END <= 1.05)}'" ""

#--- algebraic_loop_solved (Item-2): direct-feedthrough cycle, runtime fixed-point ---
AL="$("$MATLABC" -simulate "$EX/algebraic_loop_solved.mflow")"
AL_HEAD=$(printf '%s\n' "$AL" | head -1)
check "alg-loop header"        "[[ '$AL_HEAD' == 't,g,scope' ]]" ""
# g = 0.5·(1 - g) ⇒ g_∞ = 1/3.
AL_END=$(printf '%s\n' "$AL" | awk -F, 'NR>1 && $1+0>=0.9 {print $2; exit}')
check "alg-loop converges to 1/3" \
  "awk 'BEGIN{exit !(($AL_END - 0.3333333)^2 < 1e-6)}'" ""
# IR dump should still flag the loop.
AL_IR=$("$MATLABC" -simulate --dry-run "$EX/algebraic_loop_solved.mflow" 2>&1 | grep -c 'algebraic-loops')
check "alg-loop surfaced in IR" "[[ '$AL_IR' == '1' ]]" ""

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
