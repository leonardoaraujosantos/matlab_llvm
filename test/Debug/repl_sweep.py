#!/usr/bin/env python3
"""REPL (-repl) sweep over examples/ — companion to jit_parity_sweep.py.

Feeds each in-scope example to `matlabc -repl` over stdin (the file body
followed by `exit`) and classifies the outcome. Shares the exact scope /
skip rules of the DAP parity sweep so the three axes (AOT / -dap / -repl)
are comparable example-for-example.

Outcome per example:
  OK      ran to exit, no crash, no compile-error markers on stderr
  ERROR   stderr carries a compile diagnostic (error: / unsupported call
          shape / unconverted matlab.* ops / failed to compile)
  CRASH   the REPL process died with a signal (SIGSEGV …)
  HANG    no exit within the timeout
  SKIP    out of scope (same rules as jit_parity_sweep.skip_scope)

NOTE: the REPL is lenient about *runtime* semantics — an undefined name or
a dimension mismatch prints nothing and still exits 0 — so OK here means
"compiled + did not crash", not "numerically correct". The AOT lane is the
authority on run-to-exit-0; this lane adds the interactive-JIT crash/compile
signal for the ReplMode plumbing.

Usage: repl_sweep.py <matlabc> [--only SUBSTR] [--timeout S] [--quiet]
"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jit_parity_sweep as jp

EXDIR = jp.EXDIR

ERR_MARKERS = re.compile(
    r"(?:^|\n)\s*error:|unsupported call shape|unconverted matlab\.|"
    r"failed to compile|LLVMTranslationDialectInterface|"
    r"error: expected|unexpected '")


def run_one(matlabc, path, timeout):
    try:
        with open(path, "r", errors="ignore") as f:
            body = f.read()
    except OSError as e:
        return ("ERROR", f"read: {e}")
    stdin = body + "\nexit\n"
    try:
        p = subprocess.run(
            [matlabc, "-repl"],
            input=stdin, capture_output=True, text=True,
            timeout=timeout, cwd=os.path.dirname(path))
    except subprocess.TimeoutExpired:
        return ("HANG", f">{timeout}s")
    if p.returncode is not None and p.returncode < 0:
        return ("CRASH", f"signal {-p.returncode}")
    err = p.stderr or ""
    m = ERR_MARKERS.search(err)
    if m:
        # surface the first error-ish line for the report
        for ln in err.splitlines():
            if ERR_MARKERS.search("\n" + ln):
                return ("ERROR", ln.strip()[:120])
        return ("ERROR", m.group(0))
    return ("OK", "")


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: repl_sweep.py <matlabc> [--only S] [--timeout S] [--quiet]")
        return 2
    matlabc = args[0]
    only = None
    timeout = 20.0
    quiet = False
    i = 1
    while i < len(args):
        if args[i] == "--only": only = args[i + 1]; i += 2
        elif args[i] == "--timeout": timeout = float(args[i + 1]); i += 2
        elif args[i] == "--quiet": quiet = True; i += 1
        else: i += 1

    examples = []
    for dp, _, fns in os.walk(EXDIR):
        for fn in fns:
            if fn.endswith(".m"):
                examples.append(os.path.join(dp, fn))
    examples.sort()

    results = {}
    t0 = time.time()
    for path in examples:
        rel = os.path.relpath(path, EXDIR)
        if only and only not in rel:
            continue
        if jp.skip_scope(rel, path):
            results[rel] = ("SKIP", "")
            continue
        status, detail = run_one(matlabc, path, timeout)
        if status != "OK" and rel in jp.AOT_KNOWN_FAILURES:
            status, detail = "SKIP", "AOT known-failure (matched)"
        results[rel] = (status, detail)
        if not quiet and status not in ("OK", "SKIP"):
            print(f"  {status:7s} {rel}  {detail}")

    inscope = {k: v for k, v in results.items() if v[0] != "SKIP"}
    by_status = {}
    for rel, (st, _) in inscope.items():
        by_status.setdefault(st, []).append(rel)
    n_ok = len(by_status.get("OK", []))
    n_in = len(inscope)
    print(f"\n=== REPL sweep: {n_ok}/{n_in} OK "
          f"({len(results) - n_in} skipped)  in {time.time() - t0:.0f}s ===")
    for st in ("ERROR", "CRASH", "HANG"):
        lst = by_status.get(st, [])
        if lst:
            dirs = {}
            for r in lst:
                d = r.split("/")[0] if "/" in r else "(root)"
                dirs[d] = dirs.get(d, 0) + 1
            summ = ", ".join(f"{d}:{n}" for d, n in sorted(dirs.items(), key=lambda x: -x[1]))
            print(f"  {st}: {len(lst)}  [{summ}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
