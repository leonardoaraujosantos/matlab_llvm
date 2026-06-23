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

# Pin a C numeric locale. The value checks below extract and compare
# CSV columns with awk, and awk converts a field to a number through
# the locale's decimal separator. Under a comma-decimal locale (e.g.
# LC_NUMERIC=pt_BR.UTF-8) mawk reads "0.25" as 0 — it stops at the '.'
# — so row selection (`$1+0==1`) and every numeric comparison fail
# spuriously (issue #307). Forcing LC_ALL=C makes '.' the decimal
# separator for both matlabc's CSV output and awk's parsing, so the
# lane is reproducible regardless of the caller's locale.
export LC_ALL=C

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

#--- pid_block: first-class signal_pid in a closed loop -------------------
# Step ref - plant feedback -> signal_pid -> 1/(s+1) plant. The integral
# term drives steady-state error to ~0, so the plant output tracks the unit
# step. Columns: t,ref,pid,plant,scope.
PB="$("$MATLABC" -simulate "$EX/pid_block.mflow")"
PB_HEAD=$(printf '%s\n' "$PB" | head -1)
PB_PLANT=$(printf '%s\n' "$PB" | tail -1 | awk -F, '{print $4}')
check "pid_block header"     "[[ '$PB_HEAD' == 't,ref,pid,plant,scope' ]]" ""
check "pid_block tracks step" "awk 'BEGIN{exit !($PB_PLANT >= 0.98 && $PB_PLANT <= 1.02)}'" ""

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

#--- stiff_bdf (§17.5 #3): ode15s implicit BDF1 with Newton ---------------
SB="$("$MATLABC" -simulate "$EX/stiff_bdf.mflow")"
SB_HEAD=$(printf '%s\n' "$SB" | head -1)
check "stiff_bdf header" "[[ '$SB_HEAD' == 't,drive,plant,scope' ]]" ""
# Stiff plant 1/(0.0001·s+1) driven by 100·cos(t). Step h=0.1
# would explode DOPRI5 (h·|λ|=1000 >> 2.78 stability bound).
# BDF1 + Newton iteration is L-stable: plant ≈ drive (phase lag
# atan(1·0.0001) ≈ 1e-4 rad, negligible).
SB_T05=$(printf '%s\n' "$SB" | awk -F, 'NR>1 && $1+0==0.5 {print $3-$2; exit}')
check "stiff plant tracks drive at t=0.5" \
  "awk 'BEGIN{exit !(($SB_T05)^2 < 1e-2)}'" ""
SB_T2=$(printf '%s\n' "$SB" | awk -F, 'NR>1 && $1+0==2.0 {print $3-$2; exit}')
check "stiff plant tracks drive at t=2.0" \
  "awk 'BEGIN{exit !(($SB_T2)^2 < 1e-2)}'" ""

#--- per_flow_solver (§17.5 #7): per-flow maxStep override ----------------
PFS="$("$MATLABC" -simulate "$EX/per_flow_solver.mflow")"
PFS_HEAD=$(printf '%s\n' "$PFS" | head -1)
check "per_flow_solver header" \
  "[[ '$PFS_HEAD' == 't,drive,loose_plant,tight_sub/tf,tight_scope' ]]" ""
# Parent flow has maxStep=0.05; library flow has 0.005. Effective
# global step = min(0.05, 0.005) = 0.005. Over 2s, that's ~401 rows.
PFS_ROWS=$(printf '%s\n' "$PFS" | wc -l | tr -d ' ')
check "step tightened to 0.005 (≈401 rows)" \
  "[[ \$PFS_ROWS -ge 380 && \$PFS_ROWS -le 410 ]]" ""
# Per-block override surfaces in the IR dump.
PFS_IR=$("$MATLABC" -simulate --dry-run "$EX/per_flow_solver.mflow" 2>&1 | grep 'tf kind=signal_transfer_fcn')
check "tight_sub/tf carries maxStep=0.005 in IR" \
  "[[ '$PFS_IR' == *'maxStep=0.005'* ]]" ""

#--- matlab_fcn_loops (§17.5 #8): for / while loops + break ---------------
MFL="$("$MATLABC" -simulate "$EX/matlab_fcn_loops.mflow")"
MFL_HEAD=$(printf '%s\n' "$MFL" | head -1)
check "matlab_fcn_loops header" \
  "[[ '$MFL_HEAD' == 't,drive,harmonic_sum,convergence,scope_h,scope_n' ]]" ""
# At drive = 1 (t = 0.25): harmonic_sum = Σ sin(k)/k for k=1..5 =
# 0.9621742, newton_inverse_sqrt(1) = √2 = 1.4142136.
MFL_H=$(printf '%s\n' "$MFL" | awk -F, 'NR>1 && $1+0==0.25 {print $3; exit}')
MFL_N=$(printf '%s\n' "$MFL" | awk -F, 'NR>1 && $1+0==0.25 {print $4; exit}')
check "for-loop harmonic sum ≈ 0.962" \
  "awk 'BEGIN{exit !(($MFL_H - 0.962174)^2 < 1e-6)}'" ""
check "Newton with break ≈ √2" \
  "awk 'BEGIN{exit !(($MFL_N - 1.414214)^2 < 1e-6)}'" ""

#--- fir_filter (§17.5 #5): 3-tap moving-average FIR ----------------------
FF="$("$MATLABC" -simulate "$EX/fir_filter.mflow")"
FF_HEAD=$(printf '%s\n' "$FF" | head -1)
check "fir_filter header" "[[ '$FF_HEAD' == 't,step,ma3,scope' ]]" ""
# Step at t=0.1; FIR ticks at 0.05s with b=[1/3, 1/3, 1/3].
# First tick after step (t=0.15) sees one '1' in the 3-tap window
# → output = 1/3. Second tick (t=0.20) → 2/3. Third (t=0.25) → 1.0.
FF_T015=$(printf '%s\n' "$FF" | awk -F, 'NR>1 && $1+0 >= 0.149 && $1+0 < 0.151 {print $3; exit}')
check "FIR tap 1 = 1/3" \
  "awk 'BEGIN{exit !(($FF_T015 - 0.333333)^2 < 1e-6)}'" ""
FF_T020=$(printf '%s\n' "$FF" | awk -F, 'NR>1 && $1+0 >= 0.199 && $1+0 < 0.201 {print $3; exit}')
check "FIR tap 2 = 2/3" \
  "awk 'BEGIN{exit !(($FF_T020 - 0.666666)^2 < 1e-6)}'" ""
FF_T025=$(printf '%s\n' "$FF" | awk -F, 'NR>1 && $1+0 >= 0.249 && $1+0 < 0.251 {print $3; exit}')
check "FIR steady-state = 1.0" \
  "awk 'BEGIN{exit !(($FF_T025 - 1.0)^2 < 1e-6)}'" ""

#--- bouncing_ball (§17.5 #2): integrator-reset port ----------------------
BB="$("$MATLABC" -simulate "$EX/bouncing_ball.mflow")"
BB_HEAD=$(printf '%s\n' "$BB" | head -1)
check "bouncing_ball header" \
  "[[ '$BB_HEAD' == 't,vel,pos,below_floor,scope' ]]" ""
# First floor strike near t = √(2·10/9.81) ≈ 1.428. The compare's
# rising edge fires within the next step; we accept anything in
# [1.42, 1.45].
BB_HIT=$(printf '%s\n' "$BB" | awk -F, 'NR>1 && $4+0 > 0.5 {print $1; exit}')
check "ball hits floor at t≈1.43" \
  "awk 'BEGIN{exit !($BB_HIT >= 1.42 && $BB_HIT <= 1.45)}'" ""
# Post-bounce peak: (0.8·v_impact)²/(2g) ≈ 0.64·14² / 19.62 ≈ 6.4m.
BB_PEAK=$(printf '%s\n' "$BB" | awk -F, 'NR>1 && $1+0 >= 1.5 && $1+0 <= 3.0 { if($3>m)m=$3 } END {print m+0}')
check "first bounce peak ≈ 6.4m" \
  "awk 'BEGIN{exit !($BB_PEAK >= 6.2 && $BB_PEAK <= 6.6)}'" ""
# Energy diminishes — after several bounces the peak is < initial.
BB_LATE_PEAK=$(printf '%s\n' "$BB" | awk -F, 'NR>1 && $1+0 >= 4.0 && $1+0 <= 6.0 { if($3>m)m=$3 } END {print m+0}')
check "energy dissipates (later peak < 6m)" \
  "awk 'BEGIN{exit !($BB_LATE_PEAK < 6.0)}'" ""

#--- bus_signals (§17.5 #1): named-field struct wires ---------------------
BS="$("$MATLABC" -simulate "$EX/bus_signals.mflow")"
BS_HEAD=$(printf '%s\n' "$BS" | head -1)
check "bus_signals header (15 cols)" \
  "[[ '$BS_HEAD' == 't,height_src,velocity_src,temp_src,telemetry[1],telemetry[2],telemetry[3],pick_temp,pick_height' ]]" ""
# Field "temperature" lives at element 3 of the bus.
BS_TEMP=$(printf '%s\n' "$BS" | awk -F, 'NR>1 && $1+0==0.5 {print $8; exit}')
BS_TEMP_BUS=$(printf '%s\n' "$BS" | awk -F, 'NR>1 && $1+0==0.5 {print $7; exit}')
check "selector projects 'temperature'" \
  "awk 'BEGIN{exit !(($BS_TEMP - $BS_TEMP_BUS)^2 < 1e-9)}'" ""
# Field "height" at element 1, value = ramp·0.5 = 0.25 at t=0.5.
BS_H=$(printf '%s\n' "$BS" | awk -F, 'NR>1 && $1+0==0.5 {print $9; exit}')
check "selector projects 'height'" \
  "awk 'BEGIN{exit !(($BS_H - 0.25)^2 < 1e-9)}'" ""

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

#--- matrix_signals (§17.5 #9): 2-D shape signals + reshape ----------------
MS="$("$MATLABC" -simulate "$EX/matrix_signals.mflow")"
MS_HEAD=$(printf '%s\n' "$MS" | head -1)
# Column names must use the [r,c] form (not [k]) for 2-D blocks.
check "matrix_signals 2-D header columns" \
  "[[ '$MS_HEAD' == *'M2x2[1,1]'* && '$MS_HEAD' == *'M2x2[2,2]'* ]]" ""
check "matrix_signals 2-D reshape columns" \
  "[[ '$MS_HEAD' == *'r2x2[1,1]'* && '$MS_HEAD' == *'r2x2[2,2]'* ]]" ""
# Constant [1 2; 3 4] scaled by 0.5 then shifted by [10 20; 30 40] →
# [10.5 21; 31.5 42] at every time step.
MS_T0=$(printf '%s\n' "$MS" | awk -F, 'NR==2 {print $10","$11","$12","$13}')
check "scale + shift broadcasts elementwise on 2x2" \
  "[[ '$MS_T0' == '1.050000000e+01,2.100000000e+01,3.150000000e+01,4.200000000e+01' ]]" ""
# Reshape [1 2 3 4] → [[1 2];[3 4]] preserves row-major order.
MS_RS=$(printf '%s\n' "$MS" | awk -F, 'NR==2 {print $14","$15","$16","$17}')
check "reshape 1x4 → 2x2 row-major" \
  "[[ '$MS_RS' == '1.000000000e+00,2.000000000e+00,3.000000000e+00,4.000000000e+00' ]]" ""
# Dry-run dumps the inferred 2-D shape on every 2-D block, plus the
# legacy width=N tag on the 1-D vec input — confirms lowering keeps
# the per-block shape distinct from the flat element count.
MS_DRY=$("$MATLABC" -simulate --dry-run "$EX/matrix_signals.mflow" 2>&1)
check "lowering tags 2-D constant"     "[[ '$MS_DRY' == *'shape=2x2'* ]]" ""
check "lowering keeps 1-D vec as width" "[[ '$MS_DRY' == *'vec'*'width=4'* ]]" ""

#--- matlab_fcn_jit (§17.5 #8 carve-out): MLIR JIT — vector + indexing ---
# The interpreter would return 0 for both of these blocks (it
# doesn't model vector literals or multi-return helpers); a non-
# zero, analytically-correct value at t=0.25 proves the simulator
# is using its installed JIT factory.
MFJ="$("$MATLABC" -simulate "$EX/matlab_fcn_jit.mflow")"
MFJ_HEAD=$(printf '%s\n' "$MFJ" | head -1)
check "matlab_fcn_jit header" \
  "[[ '$MFJ_HEAD' == 't,drive,polar,vec_norm,scope_polar,scope_vec' ]]" ""
# At drive=1.0 (t=0.25), phase=0.5:
#   polar:    r·cos(θ) = √(1+0.25)·cos(atan2(0.5,1)) = 1.0
#             (because √(x²+y²)·cos(atan2(y,x)) ≡ x)
#   vec_norm: v=[1, 1, sin(1), cos(1)] → ‖v‖₂ = √3 ≈ 1.732
MFJ_P=$(printf '%s\n' "$MFJ" | awk -F, 'NR>1 && $1+0==0.25 {print $3; exit}')
MFJ_V=$(printf '%s\n' "$MFJ" | awk -F, 'NR>1 && $1+0==0.25 {print $4; exit}')
check "JIT multi-return polar ≈ 1.0" \
  "awk 'BEGIN{exit !(($MFJ_P - 1.0)^2 < 1e-6)}'" ""
check "JIT vector L2 ≈ √3" \
  "awk 'BEGIN{exit !(($MFJ_V - 1.7320508)^2 < 1e-6)}'" ""

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

#--- #323: --list-supported-kinds catalogue (no model file) ---------------
LK="$("$MATLABC" -simulate --list-supported-kinds 2>&1)"
# Well-formed JSON array.
check "list-kinds is a JSON array" \
  "[[ '$(printf '%s' "$LK" | head -1)' == '[' ]]" ""
# A shipped kind is supported.
check "signal_constant supported" \
  "printf '%s' \"\$LK\" | grep -q '\"kind\": \"signal_constant\", \"supported\": true'" ""
# A reserved kind is flagged unsupported.
check "signal_custom reserved" \
  "printf '%s' \"\$LK\" | grep -q '\"kind\": \"signal_custom\", \"supported\": false'" ""
# Exactly the three reserved kinds report supported:false (signal_if_action,
# signal_switch_case_action, signal_custom) — they need IDE-side container
# subsystems / a plugin ABI. From Workspace and N-D lookup are now supported.
LK_RES=$(printf '%s\n' "$LK" | grep -c '"supported": false')
check "exactly 3 reserved kinds" "[[ '$LK_RES' == '3' ]]" ""

#--- #345: state_space vector x0 + per-output C --------------------------
# Undamped oscillator A=[0 1;-1 0], x0=[2;0], C=[1 0;0 1]: analytic solution
# pos(t)=2cos t, vel(t)=-2sin t. Verifies (1) the vector IC sets vel=0 (so pos
# FALLS from 2, not rises), and (2) out2 emits the 2nd state (vel), distinct
# from out1 (pos) — before the fix both ports returned pos and vel started at 2.
SS="$("$MATLABC" -simulate "$EX/state_space_vector_ic.mflow")"
SS_HEAD=$(printf '%s\n' "$SS" | head -1)
SS_POS0=$(printf '%s\n' "$SS" | awk -F, 'NR==2{print $2}')
SS_VEL0=$(printf '%s\n' "$SS" | awk -F, 'NR==2{print $3}')
SS_POSH=$(printf '%s\n' "$SS" | awk -F, '$1+0>=0.5{print $2; exit}')
SS_VELH=$(printf '%s\n' "$SS" | awk -F, '$1+0>=0.5{print $3; exit}')
check "ss header"        "[[ '$SS_HEAD' == 't,pos,vel' ]]" ""
check "ss pos(0)=2"      "awk 'BEGIN{exit !($SS_POS0>1.999 && $SS_POS0<2.001)}'" ""
check "ss vel(0)=0"      "awk 'BEGIN{exit !($SS_VEL0>-0.001 && $SS_VEL0<0.001)}'" "vel IC must be 0, not 2"
check "ss pos falls"     "awk 'BEGIN{exit !($SS_POSH<$SS_POS0)}'" "pos must fall (2cos t), not rise"
check "ss pos(.5)≈2cos"  "awk 'BEGIN{exit !($SS_POSH>1.74 && $SS_POSH<1.77)}'" ""
check "ss vel(.5)≈-2sin" "awk 'BEGIN{exit !($SS_VELH>-0.97 && $SS_VELH<-0.95)}'" "out2 must be the velocity state"

#--- #344: signal_matlab_fcn multiple outputs ----------------------------
# function [a,b] = split(u1): a = u1+100, b = u1-100, with u1 = const 5.
# out1 (a) -> scope sa = 105; out2 (b) -> scope sb = -95. Before the fix the
# block rejected any function with >1 output; out2 could not be expressed.
MM="$("$MATLABC" -simulate "$EX/matlab_fcn_multi_output.mflow")"
MM_HEAD=$(printf '%s\n' "$MM" | head -1)
MM_A=$(printf '%s\n' "$MM" | awk -F, 'NR==2{print $2}')
MM_B=$(printf '%s\n' "$MM" | awk -F, 'NR==2{print $3}')
check "matlab_fcn multi header" "[[ '$MM_HEAD' == 't,sa,sb' ]]" ""
check "matlab_fcn out1 (a)=105"  "awk 'BEGIN{exit !(($MM_A-105)^2<1e-6)}'" ""
check "matlab_fcn out2 (b)=-95"  "awk 'BEGIN{exit !(($MM_B+95)^2<1e-6)}'" "out2 must carry the 2nd output, not out1"

#--- #343: signal_awgn — Communications AWGN channel ----------------------
# Constant input 1.0 through AWGN at snr=10 dB, signalPower=1 → additive
# N(0, σ²) with σ² = 1/10^(10/10) = 0.1. Over the run (seed-fixed,
# reproducible) the output mean tracks the input (~1.0) and the variance
# tracks σ² (~0.1). First toolbox library block via the authoring recipe.
AW="$("$MATLABC" -simulate "$EX/awgn_channel.mflow")"
AW_HEAD=$(printf '%s\n' "$AW" | head -1)
AW_MEAN=$(printf '%s\n' "$AW" | awk -F, 'NR>1{n++; s+=$2} END{print s/n}')
AW_VAR=$(printf '%s\n'  "$AW" | awk -F, 'NR>1{n++; d=$2-1; ss+=d*d} END{print ss/n}')
check "awgn header"      "[[ '$AW_HEAD' == 't,sc' ]]" ""
check "awgn mean ~ input" "awk 'BEGIN{exit !($AW_MEAN>0.95 && $AW_MEAN<1.05)}'" "noise should average out to the 1.0 input"
check "awgn var ~ 0.1"    "awk 'BEGIN{exit !($AW_VAR>0.07 && $AW_VAR<0.13)}'" "variance must track sigma^2 = 1/10^(snr/10)"

#--- #343: signal_error_rate — Communications BER sink ---------------------
# tx = constant 1, rx = a 50%-duty square wave (period 2 s). Half the symbols
# mismatch, so the running error rate converges to 0.5 over the 10 s run. The
# accumulation is once-per-major-step (in commitDigitalRegisters, not evalAll),
# so the ratio is well-formed and bounded in [0, 1].
ER="$("$MATLABC" -simulate "$EX/error_rate.mflow")"
ER_HEAD=$(printf '%s\n' "$ER" | head -1)
ER_FINAL=$(printf '%s\n' "$ER" | tail -1 | awk -F, '{print $2}')
ER_OOR=$(printf '%s\n'  "$ER" | awk -F, 'NR>1{v=$2+0; if(v<-1e-9||v>1+1e-9)bad=1} END{print bad+0}')
check "error_rate header"   "[[ '$ER_HEAD' == 't,sc' ]]" ""
check "error_rate ~ 0.5"    "awk 'BEGIN{exit !($ER_FINAL>0.45 && $ER_FINAL<0.55)}'" "50%-duty mismatch must converge to BER 0.5"
check "error_rate bounded"  "[[ '$ER_OOR' == '0' ]]" "a probability ratio must stay within [0, 1]"

#--- #343: signal_running_stats — streaming mean / variance ----------------
# A sine (amplitude 2, bias 3, 1 Hz) over exactly 10 periods. The Welford
# running mean converges to the bias (3.0); the sample variance converges to
# A²/2 = 2.0. Columns: t,sm,sv.
ST="$("$MATLABC" -simulate "$EX/running_stats.mflow")"
ST_HEAD=$(printf '%s\n' "$ST" | head -1)
ST_MEAN=$(printf '%s\n' "$ST" | tail -1 | awk -F, '{print $2}')
ST_VAR=$(printf '%s\n'  "$ST" | tail -1 | awk -F, '{print $3}')
check "running_stats header" "[[ '$ST_HEAD' == 't,sm,sv' ]]" ""
check "running mean ~ bias"  "awk 'BEGIN{exit !($ST_MEAN>2.95 && $ST_MEAN<3.05)}'" "mean of A*sin+bias over whole periods is the bias (3)"
check "running var ~ A^2/2"  "awk 'BEGIN{exit !($ST_VAR>1.90 && $ST_VAR<2.10)}'" "variance of a 2.0-amplitude sine is A^2/2 = 2"

#--- #343: signal_kalman (1-state) — constant estimation from noisy meas ---
# A scalar Kalman (A=C=1) estimates a constant 5.0 from a measurement
# corrupted by AWGN (variance ~0.1, R=0.1). The estimate converges to 5 and
# its steady-state (t>10) variance is far below the measurement variance.
# Columns: t,se,sm (estimate, measurement).
KF="$("$MATLABC" -simulate "$EX/kalman_constant.mflow")"
KF_HEAD=$(printf '%s\n' "$KF" | head -1)
KF_EST=$(printf '%s\n'  "$KF" | awk -F, 'NR>1 && $1>10{n++; s+=$2} END{print s/n}')
KF_RATIO=$(printf '%s\n' "$KF" | awk -F, 'NR>1 && $1>10{n++; se+=$2; se2+=$2*$2; sm+=$3; sm2+=$3*$3} END{ev=se2/n-(se/n)^2; mv=sm2/n-(sm/n)^2; print (mv>0)?ev/mv:1}')
check "kalman header"        "[[ '$KF_HEAD' == 't,se,sm' ]]" ""
check "kalman est ~ truth"   "awk 'BEGIN{exit !($KF_EST>4.7 && $KF_EST<5.3)}'" "estimate must converge to the constant 5.0"
check "kalman smooths noise" "awk 'BEGIN{exit !($KF_RATIO<0.25)}'" "estimate variance must be well below the measurement variance"

#--- #343: signal_kalman (2-state) — constant-velocity tracker ------------
# A 2-state Kalman (A=[1 dt;0 1], C=[1 0]) tracks a ramp (slope 0.5) from a
# noisy position measurement. It must (a) track position to ~10 at t=20,
# (b) infer the velocity ~0.5 from position changes alone (the coupling that
# exercises the 2x2 matrix recursion), (c) be smoother than the measurement.
# The vector estimate expands to two scope columns: t,se[1]=pos,se[2]=vel,truth,meas.
KT="$("$MATLABC" -simulate "$EX/kalman_tracker.mflow")"
KT_HEAD=$(printf '%s\n' "$KT" | head -1)
KT_POS=$(printf '%s\n'  "$KT" | tail -1 | awk -F, '{print $2}')
KT_VEL=$(printf '%s\n'  "$KT" | awk -F, 'NR>1 && $1>10{n++; s+=$3} END{print s/n}')
KT_SMOOTH=$(printf '%s\n' "$KT" | awk -F, 'NR>1 && $1>10{pe=$2-$4; spe+=pe*pe; me=$5-$4; sme+=me*me} END{print (spe<sme)?1:0}')
check "kalman2 header"       "[[ '$KT_HEAD' == 't,se[1],se[2],st,sm' ]]" "vector estimate must expand to per-element scope columns"
check "kalman2 tracks pos"   "awk 'BEGIN{exit !($KT_POS>9.0 && $KT_POS<11.0)}'" "position estimate must track the ramp to ~10 at t=20"
check "kalman2 infers vel"   "awk 'BEGIN{exit !($KT_VEL>0.4 && $KT_VEL<0.6)}'" "velocity state must converge to the true slope 0.5 (state coupling)"
check "kalman2 smoother"     "[[ '$KT_SMOOTH' == '1' ]]" "tracked position must be smoother than the raw measurement"

#--- #343: DSP frame trio — signal_fft / signal_ifft ----------------------
# A constant frame [1 2 3 4] → fft → ifft. The fft output (width 2N = 8) packs
# [Re_0..Re_3, Im_0..Im_3]; the round-trip ifft (width 4) must recover the
# input exactly. Columns: t, sf[1..8] (spectrum), si[1..4] (reconstruction).
FF="$("$MATLABC" -simulate "$EX/fft_roundtrip.mflow")"
FF_HEAD=$(printf '%s\n' "$FF" | head -1)
FF_LAST=$(printf '%s\n' "$FF" | tail -1)
FF_DC=$(printf '%s\n'  "$FF" | tail -1 | awk -F, '{print $2}')   # Re[0] = Σx = 10
FF_IMDC=$(printf '%s\n' "$FF" | tail -1 | awk -F, '{print $6}')  # Im[0] = 0
FF_R1=$(printf '%s\n'  "$FF" | tail -1 | awk -F, '{print $10}')  # ifft[0] = 1
FF_R4=$(printf '%s\n'  "$FF" | tail -1 | awk -F, '{print $13}')  # ifft[3] = 4
check "fft header (2N spectrum + N recon)" "[[ '$FF_HEAD' == 't,sf[1],sf[2],sf[3],sf[4],sf[5],sf[6],sf[7],sf[8],si[1],si[2],si[3],si[4]' ]]" ""
check "fft DC bin = sum"     "awk 'BEGIN{exit !(($FF_DC-10)^2<1e-6)}'" "Re[0] of [1 2 3 4] DFT is the sum, 10"
check "fft DC imag = 0"      "awk 'BEGIN{exit !($FF_IMDC^2<1e-6)}'" "Im[0] of a real frame is 0"
check "ifft recovers x[0]"   "awk 'BEGIN{exit !(($FF_R1-1)^2<1e-6)}'" "fft→ifft round-trip must recover the input frame"
check "ifft recovers x[3]"   "awk 'BEGIN{exit !(($FF_R4-4)^2<1e-6)}'" "fft→ifft round-trip must recover the input frame"

#--- #343: DSP — signal_window (Hann taper) -------------------------------
# A flat frame [1 1 1 1] through a 4-point Hann window → [0, 0.75, 0.75, 0].
# Columns: t, sw[1..4].
WW="$("$MATLABC" -simulate "$EX/window_taper.mflow")"
WW_HEAD=$(printf '%s\n' "$WW" | head -1)
WW_E1=$(printf '%s\n' "$WW" | tail -1 | awk -F, '{print $2}')  # 0
WW_E2=$(printf '%s\n' "$WW" | tail -1 | awk -F, '{print $3}')  # 0.75
check "window header"        "[[ '$WW_HEAD' == 't,sw[1],sw[2],sw[3],sw[4]' ]]" ""
check "window endpoint = 0"  "awk 'BEGIN{exit !($WW_E1^2<1e-9)}'" "Hann window is 0 at the first sample"
check "window peak = 0.75"   "awk 'BEGIN{exit !(($WW_E2-0.75)^2<1e-6)}'" "Hann(N=4) interior coefficient is 0.75"

#--- #343: signal_biquad — 2nd-order section (lowpass) --------------------
# A 2nd-order Butterworth lowpass (unity DC gain) filters a noisy constant
# (2.0 + AWGN, var ~0.1). The steady-state (t>10) output mean tracks the DC
# input (DC gain ≈ 1) and its variance is well below the input's (lowpass
# smoothing — validates the b/a difference equation, not a pass-through).
# Columns: t,sy (filtered), sm (noisy input).
BQ="$("$MATLABC" -simulate "$EX/biquad_lowpass.mflow")"
BQ_HEAD=$(printf '%s\n' "$BQ" | head -1)
BQ_MEAN=$(printf '%s\n' "$BQ" | awk -F, 'NR>1 && $1>10{n++; s+=$2} END{print s/n}')
BQ_RATIO=$(printf '%s\n' "$BQ" | awk -F, 'NR>1 && $1>10{n++; sy+=$2; sy2+=$2*$2; sm+=$3; sm2+=$3*$3} END{yv=sy2/n-(sy/n)^2; mv=sm2/n-(sm/n)^2; print (mv>0)?yv/mv:1}')
BQ_FIRST=$(printf '%s\n' "$BQ" | awk -F, 'NR==3{print ($2<0)?-$2:$2}')
check "biquad header"        "[[ '$BQ_HEAD' == 't,sy,sm' ]]" ""
check "biquad DC gain ~ 1"   "awk 'BEGIN{exit !($BQ_MEAN>1.8 && $BQ_MEAN<2.2)}'" "unity-DC-gain lowpass must pass the constant 2.0 through"
check "biquad smooths"       "awk 'BEGIN{exit !($BQ_RATIO<0.5)}'" "filtered variance must be well below the noisy input's"
check "biquad has transient" "awk 'BEGIN{exit !($BQ_FIRST<0.5)}'" "2nd-order step response rises from ~0, not a pass-through"

#--- #343: Communications — PSK modulator / demodulator -------------------
# A counter cycles symbols 0..3; QPSK-mod → demod recovers them, so the BER
# (vs the source symbol) is 0 over every symbol. A constant symbol 1 maps to
# the constellation point exp(j·π/2) = (0, 1). Columns: t,sb (BER),
# sc[1],sc[2] (I/Q of symbol 1).
PQ="$("$MATLABC" -simulate "$EX/psk_qpsk.mflow")"
PQ_HEAD=$(printf '%s\n' "$PQ" | head -1)
PQ_BERMAX=$(printf '%s\n' "$PQ" | awk -F, 'NR>1{if($2>m)m=$2} END{print m+0}')
PQ_CI=$(printf '%s\n' "$PQ" | tail -1 | awk -F, '{print ($3<0)?-$3:$3}')  # |I| ~ 0
PQ_CQ=$(printf '%s\n' "$PQ" | tail -1 | awk -F, '{print $4}')             # Q ~ 1
check "psk header"           "[[ '$PQ_HEAD' == 't,sb,sc[1],sc[2]' ]]" ""
check "psk round-trip BER=0" "awk 'BEGIN{exit !($PQ_BERMAX<1e-9)}'" "QPSK mod→demod must recover every symbol (0 errors)"
check "psk constellation I~0" "awk 'BEGIN{exit !($PQ_CI<1e-6)}'" "symbol 1 → exp(jπ/2): in-phase ≈ 0"
check "psk constellation Q~1" "awk 'BEGIN{exit !(($PQ_CQ-1)^2<1e-6)}'" "symbol 1 → exp(jπ/2): quadrature ≈ 1"

#--- #343: Communications — QAM modulator / demodulator -------------------
# A counter cycles symbols 0..15; 16-QAM mod → demod recovers them all (BER 0).
# Symbol 0 sits at the grid corner (I,Q) = (-3,-3). Columns: t,sb (BER),
# sc[1],sc[2] (I/Q of symbol 0).
QM="$("$MATLABC" -simulate "$EX/qam16.mflow")"
QM_HEAD=$(printf '%s\n' "$QM" | head -1)
QM_BERMAX=$(printf '%s\n' "$QM" | awk -F, 'NR>1{if($2>m)m=$2} END{print m+0}')
QM_CI=$(printf '%s\n' "$QM" | tail -1 | awk -F, '{print $3}')  # -3
QM_CQ=$(printf '%s\n' "$QM" | tail -1 | awk -F, '{print $4}')  # -3
check "qam header"           "[[ '$QM_HEAD' == 't,sb,sc[1],sc[2]' ]]" ""
check "qam round-trip BER=0" "awk 'BEGIN{exit !($QM_BERMAX<1e-9)}'" "16-QAM mod→demod must recover every symbol (0 errors)"
check "qam constellation corner" "awk 'BEGIN{exit !(($QM_CI+3)^2<1e-6 && ($QM_CQ+3)^2<1e-6)}'" "symbol 0 → grid corner (I,Q)=(-3,-3)"

#--- #343: vector AWGN — end-to-end noisy QPSK link -----------------------
# QPSK symbols through a *vector* AWGN channel (noises both I and Q). At high
# SNR (20 dB) the link recovers every symbol (BER 0); at low SNR (0 dB) the
# channel degrades it (BER > 0). The constant-symbol-0 tap (1,0) shows BOTH
# components carry noise — the scalar AWGN would leave Q at exactly 0.
# Columns: t, sb (BER@20), sc[1]=I, sc[2]=Q (noisy sym0), sl (BER@0).
LK="$("$MATLABC" -simulate "$EX/qpsk_awgn_link.mflow")"
LK_HEAD=$(printf '%s\n' "$LK" | head -1)
LK_BER20=$(printf '%s\n' "$LK" | awk -F, 'NR>1{if($2>m)m=$2} END{print m+0}')
LK_BER0=$(printf '%s\n'  "$LK" | tail -1 | awk -F, '{print $5}')
LK_QVAR=$(printf '%s\n'  "$LK" | awk -F, 'NR>1{n++; q+=$4; q2+=$4*$4} END{print q2/n-(q/n)^2}')
check "awgn link header"     "[[ '$LK_HEAD' == 't,sb,sc[1],sc[2],sl' ]]" ""
check "awgn link recovers @20dB" "awk 'BEGIN{exit !($LK_BER20<1e-9)}'" "high-SNR QPSK link must recover every symbol"
check "awgn link degrades @0dB"  "awk 'BEGIN{exit !($LK_BER0>0.1)}'" "low-SNR channel must cause symbol errors (BER>0)"
check "awgn noises Q component"  "awk 'BEGIN{exit !($LK_QVAR>0.05)}'" "vector AWGN must noise the quadrature, not just in-phase"

#--- #343: HDL memory — shift register / RAM / ROM -----------------------
# A 3-stage shift register (serial in = 1) is a delay line: the 1 marches one
# stage per clock and reaches the output after exactly 3 posedges (t≈1.5). A
# RAM writes data=7 at addr=2 on the clock and reads it back. A ROM maps
# addr=2 to content[2]=30. Columns: t, ss (shift out), sr (RAM), so (ROM).
MM="$("$MATLABC" -simulate "$EX/hdl_memory.mflow")"
MM_HEAD=$(printf '%s\n' "$MM" | head -1)
MM_SH_EARLY=$(printf '%s\n' "$MM" | awk -F, '$1>0.9 && $1<1.1{print $2; exit}')  # after 2 clks: 0
MM_SH_LATE=$(printf '%s\n'  "$MM" | tail -1 | awk -F, '{print $2}')              # after 3+ clks: 1
MM_RAM=$(printf '%s\n'      "$MM" | tail -1 | awk -F, '{print $3}')              # 7
MM_ROM=$(printf '%s\n'      "$MM" | tail -1 | awk -F, '{print $4}')              # 30
check "hdl_memory header"    "[[ '$MM_HEAD' == 't,ss,sr,so' ]]" ""
check "shift reg delay line" "awk 'BEGIN{exit !(($MM_SH_EARLY)^2<1e-9 && ($MM_SH_LATE-1)^2<1e-9)}'" "3-stage shift register delays the serial input by 3 clocks"
check "ram write/read"       "awk 'BEGIN{exit !(($MM_RAM-7)^2<1e-9)}'" "RAM must read back the value written at the address"
check "rom lookup"           "awk 'BEGIN{exit !(($MM_ROM-30)^2<1e-9)}'" "ROM addr 2 → content[2] = 30"

#--- #343: DSP — signal_spectrum (power spectrum |X[k]|²) -----------------
# A DC frame [1 1 1 1] concentrates all power at bin 0 (|DFT[0]|² = 4² = 16);
# a Nyquist frame [1 -1 1 -1] concentrates it at bin 2 (16). Columns:
# t, sd[1..4] (DC spectrum), sn[1..4] (Nyquist spectrum).
SP="$("$MATLABC" -simulate "$EX/spectrum.mflow")"
SP_HEAD=$(printf '%s\n' "$SP" | head -1)
SP_DC0=$(printf '%s\n'  "$SP" | tail -1 | awk -F, '{print $2}')   # DC bin power = 16
SP_DC1=$(printf '%s\n'  "$SP" | tail -1 | awk -F, '{v=$3; print (v<0)?-v:v}') # ~0
SP_NYQ=$(printf '%s\n'  "$SP" | tail -1 | awk -F, '{print $8}')   # Nyquist bin power = 16
SP_NY0=$(printf '%s\n'  "$SP" | tail -1 | awk -F, '{v=$6; print (v<0)?-v:v}') # bin0 ~0
check "spectrum header"      "[[ '$SP_HEAD' == 't,sd[1],sd[2],sd[3],sd[4],sn[1],sn[2],sn[3],sn[4]' ]]" ""
check "spectrum DC bin = 16" "awk 'BEGIN{exit !(($SP_DC0-16)^2<1e-6 && $SP_DC1<1e-6)}'" "DC frame puts all power in bin 0"
check "spectrum Nyquist bin = 16" "awk 'BEGIN{exit !(($SP_NYQ-16)^2<1e-6 && $SP_NY0<1e-6)}'" "alternating frame puts all power at the Nyquist bin"

#--- #343: Control — signal_lqr (state-feedback gain u = -K·x) ------------
# Scalar gain K=[3 4] on x=[2 1] gives u = -(3·2+4·1) = -10. A 2×2 gain
# K=[1 0; 0 2] on x=[5 3] gives the vector u = -[5; 6]. Columns: t, su (scalar
# u), sv[1],sv[2] (vector u).
LQ="$("$MATLABC" -simulate "$EX/lqr_feedback.mflow")"
LQ_HEAD=$(printf '%s\n' "$LQ" | head -1)
LQ_U=$(printf '%s\n'  "$LQ" | tail -1 | awk -F, '{print $2}')
LQ_V1=$(printf '%s\n' "$LQ" | tail -1 | awk -F, '{print $3}')
LQ_V2=$(printf '%s\n' "$LQ" | tail -1 | awk -F, '{print $4}')
check "lqr header"           "[[ '$LQ_HEAD' == 't,su,sv[1],sv[2]' ]]" ""
check "lqr scalar gain"      "awk 'BEGIN{exit !(($LQ_U+10)^2<1e-9)}'" "u = -K·x with K=[3 4], x=[2 1] is -10"
check "lqr MIMO gain"        "awk 'BEGIN{exit !(($LQ_V1+5)^2<1e-9 && ($LQ_V2+6)^2<1e-9)}'" "2×2 gain produces the vector u = -[5; 6]"

#--- #343: DSP — streaming filters (lowpass / highpass / dcblock) ---------
# One-pole streaming filters as discrete_filter presets. On a constant input:
# the lowpass passes DC (output → input, with a settling transient); the
# highpass and dcblock both reject DC (output → 0). Columns: t, slp, shp, sdc.
FL="$("$MATLABC" -simulate "$EX/streaming_filters.mflow")"
FL_HEAD=$(printf '%s\n' "$FL" | head -1)
FL_LP=$(printf '%s\n'  "$FL" | tail -1 | awk -F, '{print $2}')
FL_LP0=$(printf '%s\n' "$FL" | awk -F, '$1>0.1 && $1<0.2{print $2; exit}')  # transient < 5
FL_HP=$(printf '%s\n'  "$FL" | tail -1 | awk -F, '{v=$3; print (v<0)?-v:v}')
FL_DC=$(printf '%s\n'  "$FL" | tail -1 | awk -F, '{v=$4; print (v<0)?-v:v}')
check "filters header"       "[[ '$FL_HEAD' == 't,slp,shp,sdc' ]]" ""
check "lowpass passes DC"    "awk 'BEGIN{exit !(($FL_LP-5)^2<1e-4 && $FL_LP0<4.9)}'" "one-pole lowpass settles to the DC input (5) with a transient"
check "highpass blocks DC"   "awk 'BEGIN{exit !($FL_HP<1e-3)}'" "one-pole highpass must reject the constant (→ 0)"
check "dcblock blocks DC"    "awk 'BEGIN{exit !($FL_DC<1e-3)}'" "DC blocker must reject the constant (→ 0)"

#--- #343: Deep Learning — signal_dnn_predict (MLP inference) -------------
# One-hidden-layer MLP, y = W2·relu(W1·x + b1) + b2. Case A: x=[3 -2] through
# identity W1 + relu + W2=[1 1] gives 3 (linear would give 1 — proves the relu
# nonlinearity fires). Case B: scalar x=1 with W1=[2;-3], b1=[1 1] → hidden
# relu([3 -2]) = [3 0], 2-output identity → [3 0]. Columns: t, sa, sb[1],sb[2].
DN="$("$MATLABC" -simulate "$EX/dnn_predict.mflow")"
DN_HEAD=$(printf '%s\n' "$DN" | head -1)
DN_A=$(printf '%s\n'  "$DN" | tail -1 | awk -F, '{print $2}')
DN_B1=$(printf '%s\n' "$DN" | tail -1 | awk -F, '{print $3}')
DN_B2=$(printf '%s\n' "$DN" | tail -1 | awk -F, '{v=$4; print (v<0)?-v:v}')
check "dnn header"           "[[ '$DN_HEAD' == 't,sa,sb[1],sb[2]' ]]" ""
check "dnn relu nonlinearity" "awk 'BEGIN{exit !(($DN_A-3)^2<1e-9)}'" "relu must zero the negative pre-activation (output 3, not the linear 1)"
check "dnn scale+bias+2out"  "awk 'BEGIN{exit !(($DN_B1-3)^2<1e-9 && $DN_B2<1e-9)}'" "W1 scaling + b1 bias + relu + 2-output → [3, 0]"

#--- #343: Reinforcement Learning — signal_rl_agent (policy in the loop) --
# A trained policy maps state → action. Discrete: the MLP emits Q=[2 5 1] and
# the agent picks argmax → action index 1. Continuous: a saturating raw value
# (W2=10) through actionScale·tanh → 2·tanh(10) ≈ 2, bounded by ±actionScale.
# Columns: t, sda (discrete action index), sca (continuous action).
RL="$("$MATLABC" -simulate "$EX/rl_agent.mflow")"
RL_HEAD=$(printf '%s\n' "$RL" | head -1)
RL_D=$(printf '%s\n'  "$RL" | tail -1 | awk -F, '{print $2}')
RL_C=$(printf '%s\n'  "$RL" | tail -1 | awk -F, '{print $3}')
check "rl_agent header"      "[[ '$RL_HEAD' == 't,sda,sca' ]]" ""
check "rl discrete argmax"   "awk 'BEGIN{exit !(($RL_D-1)^2<1e-9)}'" "discrete policy must select the argmax action (index 1 of [2 5 1])"
check "rl continuous bound"  "awk 'BEGIN{exit !($RL_C>1.99 && $RL_C<=2.0)}'" "continuous policy must bound the action to ±actionScale via tanh"

#--- #343: Vision — 2-D image signals (source / filter / threshold) -------
# Grayscale images flow as flattened row-major vectors carrying their shape, so
# a scope renders 2-D [row,col] indices. A 3×3 constant-5 image through a
# normalized box filter keeps its interior (center=5, unity DC) and loses the
# zero-padded border. A vertical-edge image through Sobel-x gives a strong
# center response (4). A 2×2 ramp thresholds at 0.5 to [0 1; 0 1].
# Cols: t, sBox[1,1..3,3] ($2-$10), sSob[1,1..3,3] ($11-$19), sThr[1,1..2,2] ($20-$23).
IM="$("$MATLABC" -simulate "$EX/image_blocks.mflow")"
IM_HEAD=$(printf '%s\n' "$IM" | head -1)
IM_BOXCTR=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $6}')   # box center = 5
IM_BOXCNR=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $2}')   # box corner < 5
IM_SOBCTR=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $15}')  # sobel center = 4
IM_T1=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $20}')      # 0
IM_T2=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $21}')      # 1
IM_T4=$(printf '%s\n' "$IM" | tail -1 | awk -F, '{print $23}')      # 1
check "image 2-D shape naming" "grep -q 'sBox\[2,2\]' <<< '$IM_HEAD' && grep -q 'sThr\[2,2\]' <<< '$IM_HEAD'" "a 2-D image signal must render [row,col] scope columns"
check "box filter unity DC"  "awk 'BEGIN{exit !(($IM_BOXCTR-5)^2<1e-9 && $IM_BOXCNR<5)}'" "normalized box filter preserves the interior (5) and reduces zero-padded borders"
check "sobel edge response"  "awk 'BEGIN{exit !(($IM_SOBCTR-4)^2<1e-9)}'" "Sobel-x must respond strongly at the vertical edge"
check "threshold binarize"   "awk 'BEGIN{exit !($IM_T1<1e-9 && ($IM_T2-1)^2<1e-9 && ($IM_T4-1)^2<1e-9)}'" "per-pixel threshold at 0.5 → [0 1; 0 1]"

#--- #343: Wavelet — signal_dwt / signal_idwt (1-level Haar) --------------
# DWT of [1 2 3 4]: approx = [(1+2)/√2, (3+4)/√2] = [2.121, 4.950], detail =
# [(1-2)/√2, (3-4)/√2] = [-0.707, -0.707], packed [approx; detail]. The inverse
# recovers [1 2 3 4]. Columns: t, sd[1..4] (DWT), si[1..4] (reconstruction).
DW="$("$MATLABC" -simulate "$EX/dwt_haar.mflow")"
DW_HEAD=$(printf '%s\n' "$DW" | head -1)
DW_A0=$(printf '%s\n' "$DW" | tail -1 | awk -F, '{print $2}')   # approx[0] = 3/√2
DW_D0=$(printf '%s\n' "$DW" | tail -1 | awk -F, '{print $4}')   # detail[0] = -1/√2
DW_R1=$(printf '%s\n' "$DW" | tail -1 | awk -F, '{print $6}')   # idwt[0] = 1
DW_R4=$(printf '%s\n' "$DW" | tail -1 | awk -F, '{print $9}')   # idwt[3] = 4
check "dwt header"           "[[ '$DW_HEAD' == 't,sd[1],sd[2],sd[3],sd[4],si[1],si[2],si[3],si[4]' ]]" ""
check "dwt Haar approx"      "awk 'BEGIN{exit !(($DW_A0-2.1213203)^2<1e-6)}'" "Haar approx[0] of [1 2] is (1+2)/√2 = 2.1213"
check "dwt Haar detail"      "awk 'BEGIN{exit !(($DW_D0+0.7071068)^2<1e-6)}'" "Haar detail[0] of [1 2] is (1-2)/√2 = -0.7071"
check "dwt→idwt identity"    "awk 'BEGIN{exit !(($DW_R1-1)^2<1e-9 && ($DW_R4-4)^2<1e-9)}'" "dwt → idwt round-trip recovers the frame"

#--- #343: RF — signal_rf_2port (scattering 2-port b = S·a) ---------------
# An attenuating 2-port S=[0.1 0.9; 0.9 0.1] on incident a=[3 5] gives
# b=[0.1·3+0.9·5, 0.9·3+0.1·5]=[4.8, 3.2]; an ideal-through S=[0 1; 1 0] swaps
# the waves → [5, 3]. Columns: t, sa[1],sa[2] (attenuator), st[1],st[2] (thru).
RF="$("$MATLABC" -simulate "$EX/rf_2port.mflow")"
RF_HEAD=$(printf '%s\n' "$RF" | head -1)
RF_B1=$(printf '%s\n' "$RF" | tail -1 | awk -F, '{print $2}')
RF_B2=$(printf '%s\n' "$RF" | tail -1 | awk -F, '{print $3}')
RF_T1=$(printf '%s\n' "$RF" | tail -1 | awk -F, '{print $4}')
RF_T2=$(printf '%s\n' "$RF" | tail -1 | awk -F, '{print $5}')
check "rf_2port header"      "[[ '$RF_HEAD' == 't,sa[1],sa[2],st[1],st[2]' ]]" ""
check "rf_2port scatter"     "awk 'BEGIN{exit !(($RF_B1-4.8)^2<1e-9 && ($RF_B2-3.2)^2<1e-9)}'" "b = S·a for the attenuating 2-port is [4.8, 3.2]"
check "rf_2port ideal thru"  "awk 'BEGIN{exit !(($RF_T1-5)^2<1e-9 && ($RF_T2-3)^2<1e-9)}'" "an ideal through 2-port (anti-diagonal S) swaps the waves → [5, 3]"

#--- #343: Nav/Robotics — signal_pose_transform (2-D rigid transform) -----
# out = R(theta)·p + [x, y]. Rotating [1 0] by 90° then translating by (2,3)
# gives [0 1] + [2 3] = [2 4]; a pure translation (theta=0) of [1 1] by (5,5)
# gives [6 6]. Columns: t, sr[1],sr[2] (rotate+translate), st[1],st[2] (translate).
PT="$("$MATLABC" -simulate "$EX/pose_transform.mflow")"
PT_HEAD=$(printf '%s\n' "$PT" | head -1)
PT_R1=$(printf '%s\n' "$PT" | tail -1 | awk -F, '{print $2}')
PT_R2=$(printf '%s\n' "$PT" | tail -1 | awk -F, '{print $3}')
PT_T1=$(printf '%s\n' "$PT" | tail -1 | awk -F, '{print $4}')
PT_T2=$(printf '%s\n' "$PT" | tail -1 | awk -F, '{print $5}')
check "pose header"          "[[ '$PT_HEAD' == 't,sr[1],sr[2],st[1],st[2]' ]]" ""
check "pose rotate+translate" "awk 'BEGIN{exit !(($PT_R1-2)^2<1e-9 && ($PT_R2-4)^2<1e-9)}'" "R(90°)·[1 0] + (2,3) = [2 4]"
check "pose pure translate"  "awk 'BEGIN{exit !(($PT_T1-6)^2<1e-9 && ($PT_T2-6)^2<1e-9)}'" "θ=0 translation of [1 1] by (5,5) = [6 6]"

#--- #343: Vision — signal_color_space (RGB↔grayscale) -------------------
# rgb2gray collapses interleaved RGB triples 3→1 with luma weights 0.299/0.587/
# 0.114: red [1 0 0]→0.299, green [0 1 0]→0.587. gray2rgb expands 1→3,
# replicating: 0.5→[0.5 0.5 0.5]. Cols: t, sg[1],sg[2] (gray), sr[1..3] (rgb).
CS="$("$MATLABC" -simulate "$EX/color_space.mflow")"
CS_HEAD=$(printf '%s\n' "$CS" | head -1)
CS_R=$(printf '%s\n' "$CS" | tail -1 | awk -F, '{print $2}')   # red → 0.299
CS_G=$(printf '%s\n' "$CS" | tail -1 | awk -F, '{print $3}')   # green → 0.587
CS_X1=$(printf '%s\n' "$CS" | tail -1 | awk -F, '{print $4}')  # 0.5
CS_X3=$(printf '%s\n' "$CS" | tail -1 | awk -F, '{print $6}')  # 0.5
check "color_space header"   "[[ '$CS_HEAD' == 't,sg[1],sg[2],sr[1],sr[2],sr[3]' ]]" "rgb2gray collapses 3→1, gray2rgb expands 1→3 (widths via inference)"
check "rgb2gray luma"        "awk 'BEGIN{exit !(($CS_R-0.299)^2<1e-9 && ($CS_G-0.587)^2<1e-9)}'" "red→0.299, green→0.587 (Rec.601 luma weights)"
check "gray2rgb replicate"   "awk 'BEGIN{exit !(($CS_X1-0.5)^2<1e-9 && ($CS_X3-0.5)^2<1e-9)}'" "gray2rgb replicates the gray value across R,G,B"

#--- From Workspace / To Workspace -----------------------------------------
# signal_from_workspace replays an inline time-series ([t v; …]): linear
# interpolation (simout: 0,5,10,15,20 over t=0..2) or zero-order hold (held:
# 1 then 5 at t=1 then 9 at t=2). signal_to_workspace logs each as a named CSV
# column. Columns: t, simout (linear), held (zoh).
WS="$("$MATLABC" -simulate "$EX/workspace_io.mflow")"
WS_HEAD=$(printf '%s\n' "$WS" | head -1)
WS_LIN_MID=$(printf '%s\n' "$WS" | awk -F, '$1>0.49 && $1<0.51{print $2; exit}')  # t=0.5 → 5
WS_LIN_END=$(printf '%s\n' "$WS" | tail -1 | awk -F, '{print $2}')                # t=2 → 20
WS_ZOH_BEFORE=$(printf '%s\n' "$WS" | awk -F, '$1>0.49 && $1<0.51{print $3; exit}') # t=0.5 → 1 (held)
WS_ZOH_AFTER=$(printf '%s\n' "$WS" | awk -F, '$1>0.99 && $1<1.01{print $3; exit}')  # t=1 → 5 (stepped)
check "workspace header"     "[[ '$WS_HEAD' == 't,simout,held' ]]" "to_workspace names the CSV column after variableName"
check "from_workspace linear" "awk 'BEGIN{exit !(($WS_LIN_MID-5)^2<1e-9 && ($WS_LIN_END-20)^2<1e-9)}'" "linear interpolation of the inline time-series at t=0.5 (5) and t=2 (20)"
check "from_workspace zoh"    "awk 'BEGIN{exit !(($WS_ZOH_BEFORE-1)^2<1e-9 && ($WS_ZOH_AFTER-5)^2<1e-9)}'" "zero-order hold holds 1 until t=1, then steps to 5"

#--- signal_lookup_nd — N-D multilinear interpolation --------------------
# A 3-D table over axes [0 1]×[0 1]×[0 1] with corner values 0,1,10,11,100,…,111
# (Z[i,j,k] = 100i+10j+k). At the center (0.5,0.5,0.5) the trilinear result is
# the mean of the 8 corners = 55.5. Column: t, s.
LN="$("$MATLABC" -simulate "$EX/lookup_nd.mflow")"
LN_V=$(printf '%s\n' "$LN" | tail -1 | awk -F, '{print $2}')
check "lookup_nd trilinear"  "awk 'BEGIN{exit !(($LN_V-55.5)^2<1e-6)}'" "trilinear interpolation at the cell center is the mean of the 8 corners (55.5)"

#--- mflow-nd-signals: N-D wire signals (up to 6-D) ----------------------
# A width-24 frame reshaped to [2,3,4] flows as a rank-3 signal: the scope
# renders [i,j,k] columns (s[1,1,1]…s[2,3,4]) and the 24 values pass through
# unchanged (1…24). Reuses the existing flat buffer; only the shape generalizes.
ND="$("$MATLABC" -simulate "$EX/nd_reshape.mflow")"
ND_HEAD=$(printf '%s\n' "$ND" | head -1)
ND_FIRST=$(printf '%s\n' "$ND" | tail -1 | awk -F, '{print $2}')
ND_LAST=$(printf '%s\n'  "$ND" | tail -1 | awk -F, '{print $NF}')
ND_NCOL=$(printf '%s\n'  "$ND" | tail -1 | awk -F, '{print NF-1}')
check "nd reshape 3-D names" "grep -q 's\[1,1,1\]' <<< '$ND_HEAD' && grep -q 's\[2,3,4\]' <<< '$ND_HEAD'" "a rank-3 signal must render [i,j,k] scope columns"
check "nd reshape passthrough" "awk 'BEGIN{exit !(($ND_FIRST-1)^2<1e-9 && ($ND_LAST-24)^2<1e-9 && $ND_NCOL==24)}'" "the 24 elements pass through the reshape unchanged"

#--- mflow-nd-signals: color image as a rank-3 signal --------------------
# signal_image_source rows=2 cols=2 channels=3 → a [2,2,3] interleaved color
# signal (width 12): red/green/blue/white pixels. Subsumes the color-image
# channels residual as a special case of the N-D model.
NC="$("$MATLABC" -simulate "$EX/nd_color_image.mflow")"
NC_HEAD=$(printf '%s\n' "$NC" | head -1)
NC_NCOL=$(printf '%s\n' "$NC" | tail -1 | awk -F, '{print NF-1}')
NC_R=$(printf '%s\n'    "$NC" | tail -1 | awk -F, '{print $2}')   # red pixel R = 1
NC_G=$(printf '%s\n'    "$NC" | tail -1 | awk -F, '{print $3}')   # red pixel G = 0
check "nd color rank-3 shape" "grep -q 's\[1,1,1\]' <<< '$NC_HEAD' && grep -q 's\[2,2,3\]' <<< '$NC_HEAD'" "a color image flows as a [rows,cols,channels] rank-3 signal"
check "nd color width 12"     "awk 'BEGIN{exit !($NC_NCOL==12)}'" "2×2×3 color image = 12 interleaved elements"
check "nd color pixel data"   "awk 'BEGIN{exit !(($NC_R-1)^2<1e-9 && $NC_G^2<1e-9)}'" "first pixel is red [1,0,0]"

#--- #343: HDL sequential blocks — D flip-flop / T flip-flop / counter -----
# A 1 Hz pulse clock drives a D-FF (D=1), a free-running T-FF, and an up
# counter over 5 s. Each updates once per clock posedge (once per major step,
# not per RK4 substep). Columns: t,sff,stg,scn.
HD="$("$MATLABC" -simulate "$EX/hdl_registers.mflow")"
HD_HEAD=$(printf '%s\n' "$HD" | head -1)
HD_DFF_END=$(printf '%s\n' "$HD" | tail -1 | awk -F, '{print $2}')
HD_TFF_MEAN=$(printf '%s\n' "$HD" | awk -F, 'NR>1{n++; s+=$3} END{print s/n}')
HD_CNT_END=$(printf '%s\n' "$HD" | tail -1 | awk -F, '{print $4}')
# counter must be monotonically non-decreasing (no double-count / no reset).
HD_MONO=$(printf '%s\n' "$HD" | awk -F, 'NR>1{if($4+0 < prev-1e-9){bad=1} prev=$4} END{print bad+0}')
check "hdl header"        "[[ '$HD_HEAD' == 't,sff,stg,scn' ]]" ""
check "dff holds D=1"     "awk 'BEGIN{exit !(($HD_DFF_END-1)^2<1e-9)}'" "D flip-flop must latch and hold D"
check "tff toggles ~50%"  "awk 'BEGIN{exit !($HD_TFF_MEAN>0.30 && $HD_TFF_MEAN<0.55)}'" "T flip-flop must toggle, not stick"
check "counter ~1/clk"    "awk 'BEGIN{exit !($HD_CNT_END>=4 && $HD_CNT_END<=6)}'" "one increment per clock period (5 periods)"
check "counter monotonic" "[[ '$HD_MONO' == '0' ]]" "a clock edge must increment exactly once (not per RK4 substep)"

#--- #343: HDL JK / SR flip-flops -----------------------------------------
# A 1 Hz clock drives a JK-FF (J=K=1 → toggles every posedge, a /2 divider)
# and an SR-FF (set pulse high over [0,3), reset pulse high over [4,7)). The
# JK output toggles (mean ~0.5); the SR latches set=1 mid-set-window and
# reset=0 mid-reset-window. Columns: t,sjk,ssr.
JS="$("$MATLABC" -simulate "$EX/hdl_jk_sr.mflow")"
JS_HEAD=$(printf '%s\n' "$JS" | head -1)
JS_JK_MEAN=$(printf '%s\n' "$JS" | awk -F, 'NR>1{n++; s+=$2} END{print s/n}')
JS_JK_TRANS=$(printf '%s\n' "$JS" | awk -F, 'NR>1{v=($2>0.5)?1:0; if(NR>2 && v!=p)c++; p=v} END{print c+0}')
JS_SR_SET=$(printf '%s\n'  "$JS" | awk -F, '$1>2.49 && $1<2.51{print ($3>0.5)?1:0; exit}')
JS_SR_RST=$(printf '%s\n'  "$JS" | awk -F, '$1>5.49 && $1<5.51{print ($3>0.5)?1:0; exit}')
check "jksr header"        "[[ '$JS_HEAD' == 't,sjk,ssr' ]]" ""
check "jkff toggles ~50%"  "awk 'BEGIN{exit !($JS_JK_MEAN>0.40 && $JS_JK_MEAN<0.60)}'" "J=K=1 must toggle on every clock posedge"
check "jkff transitions"   "awk 'BEGIN{exit !($JS_JK_TRANS>=6)}'" "an 8 s / 1 Hz toggle must flip several times (not stick)"
check "srff sets"          "[[ '$JS_SR_SET' == '1' ]]" "S=1,R=0 must latch Q=1"
check "srff resets"        "[[ '$JS_SR_RST' == '0' ]]" "S=0,R=1 must clear Q=0"

#--- #343: HDL example circuits (combinational + sequential) --------------
# Combinational — half adder: SUM=A XOR B, CARRY=A AND B. A=clk/2, B=clk/1.
HA="$("$MATLABC" -simulate "$EX/hdl_half_adder.mflow")"
HA_S=$(printf '%s\n' "$HA" | awk -F, '$1+0==0.75{print $2; exit}')  # A1 B0 -> sum1
HA_C=$(printf '%s\n' "$HA" | awk -F, '$1+0==0.25{print $3; exit}')  # A1 B1 -> carry1
check "half_adder header" "[[ '$(printf '%s\n' "$HA" | head -1)' == 't,s_sum,s_carry' ]]" ""
check "half_adder SUM=A^B"   "awk 'BEGIN{exit !(($HA_S-1)^2<1e-9)}'" "A=1,B=0 ⇒ sum=1"
check "half_adder CARRY=A&B"  "awk 'BEGIN{exit !(($HA_C-1)^2<1e-9)}'" "A=1,B=1 ⇒ carry=1"
# Combinational — full adder: at t=0.25 A=B=Cin=1 ⇒ sum=1, cout=1.
FA="$("$MATLABC" -simulate "$EX/hdl_full_adder.mflow")"
FA_S=$(printf '%s\n' "$FA" | awk -F, '$1+0==0.25{print $2; exit}')
FA_C=$(printf '%s\n' "$FA" | awk -F, '$1+0==0.25{print $3; exit}')
check "full_adder sum(1,1,1)=1"  "awk 'BEGIN{exit !(($FA_S-1)^2<1e-9)}'" ""
check "full_adder cout(1,1,1)=1" "awk 'BEGIN{exit !(($FA_C-1)^2<1e-9)}'" ""
# Sequential — shift register: the input bit reaches all 3 stages (max=1 each).
SR="$("$MATLABC" -simulate "$EX/hdl_shift_register.mflow")"
SR_M=$(printf '%s\n' "$SR" | awk -F, 'NR>1{if($2>m1)m1=$2; if($3>m2)m2=$3; if($4>m3)m3=$4} END{print (m1>0.5&&m2>0.5&&m3>0.5)?1:0}')
check "shift_register bit marches q1→q2→q3" "[[ '$SR_M' == '1' ]]" "every stage must latch the shifted bit"
# Sequential — synchronous freq divider: q1:q2:q3 transition counts ≈ 2:1 each (÷2,÷4,÷8).
FD="$("$MATLABC" -simulate "$EX/hdl_freq_divider.mflow")"
FD_R=$(printf '%s\n' "$FD" | awk -F, 'NR>1{for(i=2;i<=4;i++){v=($i>0.5); if(seen[i]&&v!=p[i])tr[i]++; p[i]=v; seen[i]=1}} END{print (tr[2]>tr[3] && tr[3]>tr[4] && tr[4]>=3)?1:0}')
check "freq_divider /2 > /4 > /8" "[[ '$FD_R' == '1' ]]" "each T-FF stage halves the toggle rate"

echo "----"
echo "passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
