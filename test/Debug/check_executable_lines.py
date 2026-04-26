#!/usr/bin/env python3
# Property check: every hook line printed on stdin must point to a
# non-blank, non-comment-only line of the .m file passed as argv[1].
#
# This is the "stepping never lands on a blank line" invariant. It is
# checked in addition to the exact .expected diff so that even if a
# fixture is intentionally re-baselined, this lower-level guarantee
# still has to hold.
import sys

if len(sys.argv) != 2:
    sys.stderr.write("usage: check_executable_lines.py <source.m>\n")
    sys.exit(2)

with open(sys.argv[1], encoding="utf-8") as f:
    src = f.read().splitlines()

bad = []
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    line = int(raw)
    if line < 1 or line > len(src):
        bad.append((line, "<out of range>"))
        continue
    text = src[line - 1]
    stripped = text.lstrip(" \t")
    if not stripped or stripped[0] in ("%", "#"):
        bad.append((line, text))

if bad:
    sys.stderr.write(
        f"check_executable_lines: {len(bad)} hook line(s) point at "
        f"non-executable rows in {sys.argv[1]}:\n"
    )
    for line, text in bad:
        sys.stderr.write(f"  line {line}: {text!r}\n")
    sys.exit(1)
