#!/usr/bin/env python3
"""Deterministic JIT (-dap) parity sweep over examples/ — issue #77.

The committed run_sweep.sh only measures the AOT (-emit-llvm) path. This
harness measures the DAP-JIT axis the way #77 cares about: can `matlabc
-dap` actually *launch + run to termination* each example that the AOT
path handles?

Deterministic by construction: examples run **serially** (one DAP server
at a time), each launched by **absolute** path, with a per-example
timeout. No concurrency, so the flaky-under-load SIGSEGV noise the issue
flagged can't hide real deltas.

Outcome per example:
  OK        launch + run to `terminated`, no crash
  LAUNCH    launch/compile failed ("failed to compile program" etc.)
  HANG      no `terminated` within the timeout
  CRASH     DAP server process died with a signal (SIGSEGV …)
  SKIP      out of scope (HDL/flowchart/stateflow/sym/heavy-training;
            AOT known-failures; function-/classdef-only files with no
            runnable entry — parity is measured only vs what AOT handles)

Usage: jit_parity_sweep.py <matlabc> [--only SUBSTR] [--timeout S] [--quiet]
"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXDIR = os.path.join(ROOT, "examples")
sys.path.insert(0, HERE)
from dap_client import DapClient, DapError, initialize_and_launch


def _load_aot_known_failures():
    """Examples the AOT (-emit-llvm) lane itself can't compile/link/run.
    They are not JIT-vs-AOT divergences — this sweep only measures what
    the AOT path handles — so they must be excluded, not counted as
    LAUNCH failures. Mirrors test/Examples/known_failures.txt (the AOT
    sweep's own baseline)."""
    s = set()
    path = os.path.join(ROOT, "test", "Examples", "known_failures.txt")
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                rel = line.split("#", 1)[0].strip()
                if rel:
                    s.add(rel)
    except OSError:
        pass
    return s


AOT_KNOWN_FAILURES = _load_aot_known_failures()

# Baseline of in-scope examples that the -dap JIT lane is *known* not to
# run clean yet (today: a few non-deterministic execution-time crashes —
# residual JIT materialization races, not compile divergences). A failure
# listed here is tolerated by --gate; a failure NOT listed is a regression.
KNOWN_ISSUES_FILE = os.path.join(HERE, "jit_parity_known_issues.txt")


def _load_known_issues():
    s = set()
    try:
        with open(KNOWN_ISSUES_FILE, errors="ignore") as f:
            for line in f:
                rel = line.split("#", 1)[0].strip()
                if rel:
                    s.add(rel)
    except OSError:
        pass
    return s

SKIP_DIRS = ("hdl/", "mflow/", "mflowlink/", "stateflow/", "verilog_a/")
# Heavy training demos: documented TIMEOUTs even on the AOT lane.
SKIP_HEAVY = {
    "rl/pendulum_ddpg.m", "rl/pendulum_td3.m", "rl/cartpole_ppo.m",
    "rl/pendulum_sac.m", "rl/countdown_grpo.m",
}

def _is_definition_only(path):
    """True for a function-only / classdef-only file — its first non-blank,
    non-comment line is `function` or `classdef`. Such a file has no
    top-level script body, so there is nothing for `-dap` to launch and run
    (MATLAB can't run a `function f(...)` file without a caller either; the
    AOT lane marks these "no main entry"). A script that merely defines
    trailing local functions starts with script code, so this stays False
    for it. Handles `%{ %}` block comments."""
    try:
        in_block = False
        with open(path, "r", errors="ignore") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if in_block:
                    if s == "%}":
                        in_block = False
                    continue
                if s == "%{":
                    in_block = True
                    continue
                if s.startswith("%"):
                    continue
                tok = re.match(r"([A-Za-z_]\w*)", s)
                return bool(tok and tok.group(1) in ("function", "classdef"))
    except OSError:
        pass
    return False


def skip_scope(rel, path):
    if any(rel.startswith(d) for d in SKIP_DIRS):
        return True
    if rel.endswith("_hdl.m") or rel in SKIP_HEAVY:
        return True
    # Function-/classdef-only files have no runnable top-level entry, so
    # there is nothing for -dap to launch (AOT marks these "no main entry").
    if _is_definition_only(path):
        return True
    # AOT-known-failures are handled post-run: a JIT failure there is a
    # matched non-divergence (-> SKIP), but a JIT *success* is a genuine
    # win worth keeping visible (-> OK). See the runner below.
    try:
        with open(path, "r", errors="ignore") as f:
            head = "".join(f.readline() for _ in range(20))
        if re.search(r"^\s*%\s*hdl:", head, re.M):
            return True
        # symbolic examples need SymPP linked; skip in this env.
        if re.search(r"(^|[^\w])(syms|sym)\s*[(]|^\s*syms\s", head, re.M):
            return True
        # interactive examples (keyboard/input/pause-for-key) can't run
        # headless — they block waiting for stdin. Read the whole file.
        with open(path, "r", errors="ignore") as f:
            body = f.read()
        if re.search(r"(^|[^\w])(keyboard|input)\s*[(;]", body, re.M):
            return True
    except OSError:
        pass
    return False

def run_one(matlabc, path, timeout):
    client = DapClient(matlabc, path)
    try:
        try:
            initialize_and_launch(client, stop_on_entry=False)
        except DapError as e:
            return ("LAUNCH", str(e).splitlines()[0][:120])
        try:
            client.wait_event("terminated", timeout=timeout)
        except DapError:
            rc = client.proc.poll()
            if rc is not None and rc < 0:
                return ("CRASH", f"signal {-rc}")
            return ("HANG", f">{timeout}s")
        rc = client.proc.poll()
        if rc is not None and rc < 0:
            return ("CRASH", f"signal {-rc}")
        return ("OK", "")
    except Exception as e:  # noqa
        return ("LAUNCH", f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")
    finally:
        try:
            client.close()
        except Exception:
            pass

def main():
    args = sys.argv[1:]
    if not args:
        print("usage: jit_parity_sweep.py <matlabc> [--only S] [--timeout S] [--quiet]")
        return 2
    matlabc = args[0]
    only = None
    timeout = 15.0
    quiet = False
    gate = False
    i = 1
    while i < len(args):
        if args[i] == "--only": only = args[i+1]; i += 2
        elif args[i] == "--timeout": timeout = float(args[i+1]); i += 2
        elif args[i] == "--quiet": quiet = True; i += 1
        elif args[i] == "--gate": gate = True; i += 1
        else: i += 1
    known_issues = _load_known_issues()

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
        if skip_scope(rel, path):
            results[rel] = ("SKIP", "")
            continue
        status, detail = run_one(matlabc, path, timeout)
        # CRASH/HANG are non-deterministic (JIT exec races). In --gate mode
        # retry once so a transient flake on an otherwise-clean example
        # doesn't turn the CI gate red; a deterministic LAUNCH (compile)
        # failure is not retried.
        if gate and status in ("CRASH", "HANG"):
            status, detail = run_one(matlabc, path, timeout)
        # AOT-known-failures aren't parity divergences. If the JIT also
        # fails one, reclassify as SKIP (matched — not a JIT bug). If the
        # JIT *succeeds*, keep the OK: it's a genuine JIT-beyond-AOT win.
        if status != "OK" and rel in AOT_KNOWN_FAILURES:
            status, detail = "SKIP", "AOT known-failure (matched)"
        results[rel] = (status, detail)
        if not quiet and status != "OK":
            print(f"  {status:7s} {rel}  {detail}")

    inscope = {k: v for k, v in results.items() if v[0] != "SKIP"}
    by_status = {}
    for rel, (st, _) in inscope.items():
        by_status.setdefault(st, []).append(rel)
    n_ok = len(by_status.get("OK", []))
    n_in = len(inscope)
    print(f"\n=== DAP parity sweep: {n_ok}/{n_in} OK "
          f"({len(results)-n_in} skipped)  in {time.time()-t0:.0f}s ===")
    for st in ("LAUNCH", "CRASH", "HANG"):
        lst = by_status.get(st, [])
        if lst:
            # group by top dir
            dirs = {}
            for r in lst:
                d = r.split("/")[0] if "/" in r else "(root)"
                dirs[d] = dirs.get(d, 0) + 1
            summ = ", ".join(f"{d}:{n}" for d, n in sorted(dirs.items(), key=lambda x: -x[1]))
            print(f"  {st}: {len(lst)}  [{summ}]")

    if not gate:
        return 0

    # --- Regression gate -------------------------------------------------
    # Any in-scope non-OK example that isn't a documented known-issue is a
    # regression. Known-issues that now pass are STALE (prune them) — a
    # warning, not a failure. Mirrors test/Examples/run_sweep.sh.
    failed = {rel: st for rel, (st, _) in inscope.items() if st != "OK"}
    regressions = sorted(r for r in failed if r not in known_issues)
    stale = sorted(r for r in known_issues
                   if r not in failed and (not only or only in r))
    if stale:
        print("\nSTALE jit_parity_known_issues.txt entries (now pass — prune):")
        for r in stale:
            print(f"  {r}")
    if regressions:
        print("\nREGRESSION — in-scope examples newly failing under -dap "
              "(not in jit_parity_known_issues.txt):")
        for r in regressions:
            print(f"  {failed[r]:7s} {r}  {results[r][1]}")
        print(f"\n=== GATE FAILED: {len(regressions)} regression(s) ===")
        return 1
    print("\n=== GATE OK: no -dap parity regressions ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
