#!/usr/bin/env python3
"""Perf-bench regression gate.

Takes two JSON result files (typically PR-HEAD vs main) and:
  1. Prints a Markdown comparison table for the matlab_llvm impl only.
  2. Exits non-zero if any matlab_llvm kernel regressed by more than
     THRESHOLD (default 30%) compared to the base.

The threshold defaults to 30% because CI runner variance on shared
GitHub Actions hosts is ~10-15% even with min-of-3 trials; tighter
thresholds produce false positives.  Override with --threshold for
local strict checks.

A relative threshold alone false-positives on the sub-2ms kernels
(chol/solve/matmul ~1ms), where absolute scheduler / CPU-frequency
jitter dominates: 30% of 1ms is 0.3ms, smaller than the run-to-run
noise (measured ~55% spread on identical code).  So a move must also
clear an ABSOLUTE floor (--min-abs-ms, default 1.0ms) to count as a
regression or improvement; smaller moves are reported but classified
as below the noise floor.  This keeps the relative gate effective for
the larger kernels and for genuinely large small-kernel regressions
(1ms→3ms still trips: +2ms > floor AND +200% > threshold).

Usage:
    check_regression.py --base <base.json> --head <head.json>
                        [--threshold 0.30] [--min-abs-ms 1.0]

Designed to be invoked by .github/workflows/ci.yml's perf-bench lane.
Tier 7 of docs/acceleration_roadmap.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())


def index(payload: dict) -> dict:
    return {(r["kernel"], r["N"], r["impl"]): r["seconds"]
            for r in payload["results"]}


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.2f}ms"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path,
                        help="Baseline results JSON (typically main).")
    parser.add_argument("--head", required=True, type=Path,
                        help="PR HEAD results JSON.")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Relative regression threshold (default 0.30 = "
                             "30%%).")
    parser.add_argument("--min-abs-ms", type=float, default=1.0,
                        help="Absolute floor in milliseconds (default 1.0). A "
                             "move must clear BOTH this and --threshold to "
                             "count; smaller moves are below the noise floor.")
    parser.add_argument("--allow-regression", action="store_true",
                        help="Print the table but exit 0 even on regression "
                             "(use for the 'perf-allowed' PR label).")
    parser.add_argument("--output", type=Path, default=None,
                        help="If set, write the Markdown table to this file "
                             "(in addition to stdout).")
    args = parser.parse_args()

    base = index(load(args.base))
    head = index(load(args.head))

    lines = ["# Perf-bench: regression check",
             "",
             f"- base: `{args.base.name}`",
             f"- head: `{args.head.name}`",
             f"- threshold: ±{args.threshold:.0%}",
             f"- noise floor: ±{args.min_abs_ms:.2f}ms (moves below this are "
             "not gated)",
             "",
             "| Kernel | N | base | head | Δ% | status |",
             "|---|---|---|---|---|---|"]

    regressions: list[str] = []
    improvements: list[str] = []
    for key in sorted(set(base.keys()) | set(head.keys())):
        kernel, N, impl = key
        if impl != "matlab_llvm":
            continue
        b = base.get(key)
        h = head.get(key)
        if b is None or h is None:
            lines.append(
                f"| `{kernel}` | {N} | {fmt(b)} | {fmt(h)} | — | (new/removed) |")
            continue
        if b <= 0 or h <= 0:
            continue
        delta = (h - b) / b
        abs_ms = abs(h - b) * 1000.0
        below_floor = abs_ms < args.min_abs_ms
        status = "OK"
        flag = ""
        if delta > args.threshold:
            if below_floor:
                status = "OK (noise floor)"
                flag = "⚪"
            else:
                status = "**REGRESSION**"
                flag = "🔴"
                regressions.append(f"{kernel} N={N}: {b * 1000:.2f}ms → "
                                   f"{h * 1000:.2f}ms ({delta:+.1%})")
        elif delta < -args.threshold:
            if below_floor:
                status = "OK (noise floor)"
                flag = "⚪"
            else:
                status = "improvement"
                flag = "🟢"
                improvements.append(f"{kernel} N={N}: {b * 1000:.2f}ms → "
                                    f"{h * 1000:.2f}ms ({delta:+.1%})")
        lines.append(
            f"| `{kernel}` | {N} | {fmt(b)} | {fmt(h)} | "
            f"{delta:+.1%} | {flag} {status} |")

    lines.append("")
    if regressions:
        lines.append(f"## ❌ {len(regressions)} regression"
                     f"{'s' if len(regressions) != 1 else ''}")
        for r in regressions:
            lines.append(f"- {r}")
        lines.append("")
    if improvements:
        lines.append(f"## ✅ {len(improvements)} improvement"
                     f"{'s' if len(improvements) != 1 else ''}")
        for r in improvements:
            lines.append(f"- {r}")
        lines.append("")
    if not regressions and not improvements:
        lines.append("No kernels moved beyond the ±{:.0%} threshold.".format(
            args.threshold))

    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")

    # GitHub Actions step-summary integration.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")

    if regressions and not args.allow_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
