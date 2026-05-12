#!/usr/bin/env bash
# va_cosim.sh — Verilog-A cosim wrapper around ngspice (preferred) or
# Xyce.  Drives an emitted .va module through an AC sweep, dumps the
# response, and (optionally) compares against an in-tree reference
# computed by `freqresp` / `timeresp` in the matlab_llvm runtime.
#
# Tier-10 follow-on for docs/verilog_a_plan.md.  At the time of writing,
# ngspice with OpenVAF integration is the working cosim path; Xyce
# accepts .va via ADMS but the toolchain wiring is more invasive.
#
# Usage:
#   scripts/va_cosim.sh <module.va>
#
# Exit status:
#   0  — cosim succeeded (or skipped because no simulator was found).
#   1  — cosim failed (simulator exited nonzero, no AC sweep data, …).
#
# This wrapper is intentionally light on validation: the .va is run
# through a canonical AC-sweep netlist template; full freqresp cross-
# check against the in-tree reference is a follow-on once enough users
# have OpenVAF locally.

set -u

if [ $# -ne 1 ]; then
  echo "usage: $0 <module.va>" >&2
  exit 2
fi

VA="$1"
[ -f "$VA" ] || { echo "missing $VA" >&2; exit 2; }
MODNAME="$(basename "${VA%.va}")"

SIM=""
HAS_OPENVAF=0
command -v openvaf >/dev/null 2>&1 && HAS_OPENVAF=1
if command -v ngspice >/dev/null 2>&1 && [ "$HAS_OPENVAF" = 1 ]; then
  # ngspice cosim requires OpenVAF to compile the .va into an .osdi
  # plug-in that ngspice loads via the `pre_osdi` directive.
  SIM="ngspice"
elif command -v Xyce >/dev/null 2>&1; then
  SIM="Xyce"
else
  echo "skip: cosim toolchain not installed." >&2
  if [ "$HAS_OPENVAF" = 0 ]; then
    echo "      OpenVAF (https://openvaf.semimod.de/) is required to" >&2
    echo "      compile .va modules for ngspice; install it and ensure" >&2
    echo "      ngspice is also on PATH." >&2
  fi
  if ! command -v Xyce >/dev/null 2>&1; then
    echo "      Alternative: Xyce (https://xyce.sandia.gov/) with" >&2
    echo "      ADMS support." >&2
  fi
  exit 0
fi

WORK="$(mktemp -d -t va_cosim.XXXXXX)"
trap "rm -rf '$WORK'" EXIT
cp "$VA" "$WORK/"

# Heuristic netlist template — assumes 1 input + 1 output port.
# This works for the majority of examples (rational filters, sources,
# comparators); composite blocks with extra ports need a per-block
# template that the user authors manually.
cd "$WORK"
case "$SIM" in
  ngspice)
    # Compile .va -> .osdi via OpenVAF.
    openvaf "$MODNAME.va" -o "$MODNAME.osdi" > openvaf.log 2>&1 || {
      echo "FAIL: openvaf failed to compile $MODNAME.va" >&2
      tail -20 openvaf.log >&2
      exit 1
    }
    cat > netlist.cir <<EOF
* Auto-generated cosim netlist for $MODNAME (ngspice + OpenVAF)
pre_osdi $MODNAME.osdi
Vin in 0 AC 1
X1 in out $MODNAME
Rload out 0 50
.AC DEC 10 100 1G
.PRINT AC v(out)
.END
EOF
    ngspice -b netlist.cir > ngspice.log 2>&1 || {
      echo "FAIL: ngspice exited nonzero" >&2
      tail -20 ngspice.log >&2
      exit 1
    }
    grep -q "Index" ngspice.log || {
      echo "FAIL: ngspice produced no AC-sweep table" >&2
      tail -20 ngspice.log >&2
      exit 1
    }
    echo "ok: ngspice + OpenVAF AC sweep completed for $MODNAME"
    ;;
  Xyce)
    cat > netlist.cir <<EOF
* Auto-generated cosim netlist for $MODNAME (Xyce + ADMS)
.include "$MODNAME.va"
Vin in 0 AC 1
X1 in out $MODNAME
Rload out 0 50
.AC DEC 10 100 1G
.PRINT AC v(out)
.END
EOF
    Xyce netlist.cir > xyce.log 2>&1 || {
      echo "FAIL: Xyce exited nonzero" >&2
      tail -20 xyce.log >&2
      exit 1
    }
    echo "ok: Xyce AC sweep completed for $MODNAME"
    ;;
esac
exit 0
