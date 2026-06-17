#!/usr/bin/env python3
"""Regression: `matlabc`'s usage string must advertise `-simulate`
(and the `--sim-dap` live-DAP transport).

Issue #310: the usage string omitted `-simulate` entirely, so the
one-shot CSV / dry-run / live-DAP simulation modes were undiscoverable
from `matlabc --help`-style output. The simulation lane is a top-level
mode and must appear in the usage banner alongside the other modes.

Drives `matlabc` with no FILE argument (which prints usage to stderr and
exits non-zero) and asserts the banner mentions `-simulate` and
`--sim-dap`.

Usage: run_usage.py <path-to-matlabc>
"""
from __future__ import annotations

import subprocess
import sys


def main(matlabc: str) -> int:
    # No FILE argument → usage() to stderr, exit 64 (EX_USAGE).
    proc = subprocess.run(
        [matlabc], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    banner = (proc.stdout + proc.stderr).decode(errors="replace")

    failures: list[str] = []
    for needle in ("-simulate", "--sim-dap"):
        if needle in banner:
            print(f"PASS  usage advertises {needle}")
        else:
            failures.append(needle)
            print(f"FAIL  usage banner missing {needle!r}")

    if proc.returncode == 0:
        failures.append("exit")
        print(f"FAIL  expected non-zero exit on missing FILE, got 0")
    else:
        print(f"PASS  non-zero exit on missing FILE: {proc.returncode}")

    print("----")
    if failures:
        print(f"FAIL  {len(failures)} check(s)")
        print("--- banner was ---")
        print(banner)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <matlabc>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
