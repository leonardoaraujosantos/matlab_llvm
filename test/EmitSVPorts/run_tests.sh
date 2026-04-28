#!/usr/bin/env bash
# fi-spec ↔ SV declaration regression suite.
#
# Each test/EmitSVPorts/<name>.m is a small synthesizable function
# whose ports declare a known fi spec via either a `% hdl: port(...)`
# pragma or a typed driver call. The matching <name>.expected file
# lists substring matches that must appear in the emitted SV — one
# per line. The suite passes when every substring appears at least
# once in the SV output.
#
# This is intentionally substring-based (not full golden diff) so
# the assertions stay focused on the property under test
# (signedness + bit width on port and register declarations) and
# don't have to be regenerated whenever some unrelated emitter
# detail changes.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.expected"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .expected)"; continue; }

  out=$(mktemp -t emsvp.XXXXXX); err=$(mktemp -t emsvp.XXXXXX)
  "$MATLABC" -emit-systemverilog "$m" >"$out" 2>"$err"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $base [emit]: matlabc exited $rc"
    sed 's/^/    /' "$err"
    fail=$((fail+1))
    rm -f "$out" "$err"
    continue
  fi

  ok=1
  missing=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    if ! grep -qF -- "$line" "$out"; then
      missing+=("$line")
      ok=0
    fi
  done < "$exp"

  if [[ $ok -eq 1 ]]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: expected substring(s) not in emitted SV"
    for line in "${missing[@]}"; do
      echo "  missing: $line"
    done
    echo "  actual port lines:"
    grep -E "^\s*(input|output|logic)" "$out" | sed 's/^/    /'
  fi
  rm -f "$out" "$err"
done

echo "----"
echo "emit-sv-ports passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
