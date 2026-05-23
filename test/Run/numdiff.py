#!/usr/bin/env python3
"""Tolerance-aware comparison for the Run goldens.

Compares two text files token by token: numeric tokens match within a
relative+absolute tolerance, every other token must match exactly.  This
absorbs last-significant-digit floating-point divergence between libm
implementations (macOS libc++ / libm vs Linux libstdc++ / glibc) that made
the `disp`-printed numeric goldens platform-dependent, while still catching
real regressions (default relative tolerance 1e-4).

Usage:  numdiff.py EXPECTED ACTUAL
Exit 0 on match, 1 on mismatch (the mismatching lines are printed), 2 on
usage error.  Tunable via env: NUMDIFF_RTOL (default 1e-4),
NUMDIFF_ATOL (default 1e-6).
"""
import os
import sys

RTOL = float(os.environ.get("NUMDIFF_RTOL", "1e-4"))
ATOL = float(os.environ.get("NUMDIFF_ATOL", "1e-6"))


def parse_num(tok):
    """Return the float value of a numeric token, or None if it isn't one.

    Handles plain reals (incl. scientific notation) and the imaginary part
    of a complex token printed as `<num>i` (e.g. `2.5i`, `-3i`)."""
    try:
        return float(tok)
    except ValueError:
        pass
    if len(tok) > 1 and tok.endswith("i"):
        try:
            return float(tok[:-1])
        except ValueError:
            return None
    return None


def tokens_match(a, b):
    if a == b:
        return True
    na, nb = parse_num(a), parse_num(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= ATOL + RTOL * max(abs(na), abs(nb))


def lines_match(la, lb):
    ta, tb = la.split(), lb.split()
    if len(ta) != len(tb):
        return False
    return all(tokens_match(x, y) for x, y in zip(ta, tb))


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: numdiff.py EXPECTED ACTUAL\n")
        return 2
    with open(sys.argv[1]) as f:
        exp = f.read().splitlines()
    with open(sys.argv[2]) as f:
        got = f.read().splitlines()
    fail = False
    for i in range(max(len(exp), len(got))):
        a = exp[i] if i < len(exp) else "<missing line>"
        b = got[i] if i < len(got) else "<missing line>"
        if i >= len(exp) or i >= len(got) or not lines_match(a, b):
            print("line %d:" % (i + 1))
            print("  - %s" % a)
            print("  + %s" % b)
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
