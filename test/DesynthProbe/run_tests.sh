#!/usr/bin/env bash
# #191 P3 — dispatch-desynth coverage contract.
#
# Compiles each fixture with MATLAB_LLVM_PROBE_LATE_MONO set and inspects the
# probe lines the lowering emits when a class operator reaches the SYNTHESIS
# fallback (i.e. one the Sema-time desynth pass did NOT rewrite). The contract:
#
#   * NO migrated class (tf/ss/zpk/pid/frd/Vec2) operator with the object on
#     the LHS (lhs_obj=1) may reach synthesis — desynth must have rewritten it.
#     A fire here is a coverage regression.
#   * The probe must still report SOMETHING (a scalar-LHS `k*G` case, the
#     deferred gap) — a liveness check so the test can't pass vacuously if the
#     probe silently stops emitting.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
MIGRATED='(tf|ss|zpk|pid|frd|Vec2)'
pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"

  tmperr="$(mktemp -t mlc.XXXXXX).err"
  MATLAB_LLVM_PROBE_LATE_MONO=1 "$MATLABC" -emit-llvm "$m" >/dev/null 2>"$tmperr"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $base: matlabc -emit-llvm exited $rc"
    sed 's/^/  /' "$tmperr" | head -8
    fail=$((fail+1)); rm -f "$tmperr"; continue
  fi

  probes="$(grep 'late-mono-probe' "$tmperr" || true)"

  # Liveness: the probe must have reported at least one line.
  if [[ -z "$probes" ]]; then
    echo "FAIL $base: probe emitted nothing (is MATLAB_LLVM_PROBE_LATE_MONO wired?)"
    fail=$((fail+1)); rm -f "$tmperr"; continue
  fi

  # Contract: no migrated-class operator with the object on the LHS may have
  # fallen through to synthesis.
  leaked="$(printf '%s\n' "$probes" \
            | grep -E "op-synth ${MIGRATED}::[a-z]+ lhs_obj=1" || true)"
  if [[ -n "$leaked" ]]; then
    echo "FAIL $base: migrated-class operator(s) reached lowering synthesis:"
    printf '%s\n' "$leaked" | sed 's/^/    /'
    fail=$((fail+1)); rm -f "$tmperr"; continue
  fi

  pass=$((pass+1))
  rm -f "$tmperr"
done

echo "----"
echo "desynth-probe passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
