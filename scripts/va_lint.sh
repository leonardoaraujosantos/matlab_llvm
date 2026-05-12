#!/usr/bin/env bash
# va_lint.sh — Verilog-A lint wrapper around OpenVAF (or ADMS as fallback).
#
# Usage:
#   scripts/va_lint.sh examples/verilog_a/*.va
#   scripts/va_lint.sh path/to/file.va
#
# Exit status: number of files that failed lint.
# Requires either `openvaf` (https://openvaf.semimod.de/) or `adms`
# (Synopsys/Cadence ADMS frontend) on PATH.  Both are optional — if
# neither is installed, the script prints a hint and exits 0 (skip).
#
# The script does not modify the .va files; it only reads them and
# reports diagnostics.

set -u

if [ $# -eq 0 ]; then
  echo "usage: $0 <file.va> [file2.va ...]" >&2
  exit 2
fi

LINTER=""
if command -v openvaf >/dev/null 2>&1; then
  LINTER="openvaf"
elif command -v adms >/dev/null 2>&1; then
  LINTER="adms"
else
  echo "skip: neither openvaf nor adms is on PATH.  Install OpenVAF" >&2
  echo "      (https://openvaf.semimod.de/) to enable Verilog-A lint." >&2
  exit 0
fi

fail=0
for f in "$@"; do
  [ -f "$f" ] || { echo "SKIP $f (not found)"; continue; }
  echo "== $LINTER: $f =="
  case "$LINTER" in
    openvaf)
      if ! openvaf "$f"; then fail=$((fail+1)); fi
      ;;
    adms)
      if ! adms -e va.xml "$f"; then fail=$((fail+1)); fi
      ;;
  esac
done

if [ $fail -eq 0 ]; then
  echo "all clean"
else
  echo "$fail file(s) failed lint" >&2
fi
exit $fail
