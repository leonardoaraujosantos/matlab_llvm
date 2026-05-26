#!/usr/bin/env python3
"""Render a before/after comparison table from two JSON result sets.

Usage:
    python3 bench/lapack/report.py <tag_before> <tag_after>

Picks results/<tag>.json and emits Markdown.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(tag: str) -> dict:
    path = HERE / "results" / f"{tag}.json"
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())


def key(r):
    return (r["kernel"], r["N"], r["impl"])


def fmt(seconds):
    if seconds is None:
        return "—"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.2f}ms"


def speedup(before, after):
    if before is None or after is None or after == 0:
        return "—"
    return f"{before / after:.2f}×"


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("usage: report.py <tag_before> <tag_after>")
    tag_before, tag_after = sys.argv[1], sys.argv[2]
    before = load(tag_before)
    after = load(tag_after)
    before_map = {key(r): r["seconds"] for r in before["results"]}
    after_map = {key(r): r["seconds"] for r in after["results"]}

    kernels = list(dict.fromkeys(r["kernel"] for r in after["results"]))
    sizes = sorted({r["N"] for r in after["results"]})

    print(f"# LAPACK acceleration: `{tag_before}` → `{tag_after}`\n")
    print(f"Host: {after['host']['os']}/{after['host']['arch']}.\n")

    for kernel in kernels:
        print(f"## `{kernel}`\n")
        print("| N | matlab_llvm (before → after) | NumPy | pure Python | Speedup vs before | matlab_llvm vs NumPy (after) |")
        print("|---|---|---|---|---|---|")
        for N in sizes:
            mb = before_map.get((kernel, N, "matlab_llvm"))
            ma = after_map.get((kernel, N, "matlab_llvm"))
            np_ = after_map.get((kernel, N, "numpy"))
            pp = after_map.get((kernel, N, "pure_python"))
            row_speedup = speedup(mb, ma)
            vs_numpy = speedup(np_, ma)  # >1 means matlab_llvm is faster
            print(
                f"| {N} | {fmt(mb)} → {fmt(ma)} | {fmt(np_)} | "
                f"{fmt(pp)} | {row_speedup} | {vs_numpy} |"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
