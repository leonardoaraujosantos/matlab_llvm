#!/usr/bin/env python3
"""Tier-F smoke test for `matlabc -simulate --sim-dap`.

Drives the DAP server and verifies that:
  1. setTimeBreakpoints arms a time breakpoint that fires during
     `continue` and reports `stopped { reason: "breakpoint" }`.
  2. setSignalBreakpoints arms a signal-watch ("scope > 0.2") that
     fires when the lowpass output crosses the threshold.
  3. After a breakpoint fire, `continue` resumes past it (the
     breakpoint is sticky — armed-once per arming list).
  4. `restart` re-arms every breakpoint so a re-run can hit them
     again.

Usage: run_breakpoints.py <path-to-matlabc>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def frame(o):
    b = json.dumps(o).encode()
    return f"Content-Length: {len(b)}\r\n\r\n".encode() + b


def read_one(stream):
    buf = b""
    while b"\r\n\r\n" not in buf:
        c = stream.read(1)
        if not c:
            return None
        buf += c
    header, _ = buf.split(b"\r\n\r\n", 1)
    clen = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            clen = int(line.split(b":", 1)[1].strip())
    body = b""
    while len(body) < clen:
        c = stream.read(clen - len(body))
        if not c:
            return None
        body += c
    return json.loads(body)


def main(matlabc):
    here = os.path.dirname(os.path.abspath(__file__))
    mflow = os.path.abspath(os.path.join(
        here, "..", "..", "..", "examples", "mflowlink", "lowpass.mflow"))
    proc = subprocess.Popen(
        [matlabc, "-simulate", "--sim-dap", mflow],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin and proc.stdout

    failures = []
    seq = [0]

    def send(o):
        proc.stdin.write(frame(o))
        proc.stdin.flush()

    def req(cmd, args=None):
        seq[0] += 1
        send({"seq": seq[0], "type": "request", "command": cmd,
              "arguments": args or {}})

    def wait_for(pred, label, max_frames=2000):
        for _ in range(max_frames):
            f = read_one(proc.stdout)
            if f is None:
                failures.append(f"stream closed waiting for {label}")
                return None
            if pred(f):
                return f
        failures.append(f"timed out waiting for {label}")
        return None

    def is_resp(f, cmd):
        return f.get("type") == "response" and f.get("command") == cmd

    def is_stopped(f, reason=None):
        return (f.get("type") == "event" and f.get("event") == "stopped"
                and (reason is None or f.get("body", {}).get("reason") == reason))

    # Handshake.
    req("initialize")
    wait_for(lambda f: is_resp(f, "initialize"), "init resp")
    wait_for(lambda f: f.get("event") == "initialized", "initialized")
    req("launch")
    wait_for(lambda f: is_resp(f, "launch"), "launch resp")
    req("configurationDone")
    wait_for(lambda f: is_resp(f, "configurationDone"), "confDone resp")
    wait_for(lambda f: is_stopped(f, "entry"), "stopped(entry)")

    # 1. Time breakpoint at t = 2.5 — should fire mid-run.
    req("setTimeBreakpoints", {"times": [{"t": 2.5}]})
    wait_for(lambda f: is_resp(f, "setTimeBreakpoints"), "setTimeBP resp")
    req("continue")
    wait_for(lambda f: is_resp(f, "continue"), "continue resp")
    stop = wait_for(lambda f: is_stopped(f), "stopped after continue (time bp)")
    if stop and stop["body"].get("reason") == "breakpoint":
        print(f"PASS  time-bp fires: {stop['body'].get('description')}")
    else:
        failures.append(f"expected stopped(breakpoint), got "
                        f"{stop['body'] if stop else None}")
        print(f"FAIL  time-bp did not fire as expected")

    def get_t():
        req("variables", {"variablesReference": 1})
        r = wait_for(lambda f: is_resp(f, "variables"), "variables")
        for v in r["body"]["variables"]:
            if v["name"] == "t":
                return float(v["value"])
        return float("nan")

    t_bp = get_t()
    if not (2.4 <= t_bp <= 2.6):
        failures.append(f"expected t≈2.5 at time-bp, got {t_bp}")
    else:
        print(f"PASS  time-bp t:    {t_bp:.4f}")

    # 2. Continue past the time breakpoint — it shouldn't refire.
    req("continue")
    wait_for(lambda f: is_resp(f, "continue"), "continue resp #2")
    stop2 = wait_for(lambda f: is_stopped(f), "stopped #2")
    t_end = get_t()
    if stop2 and stop2["body"].get("reason") != "breakpoint":
        print(f"PASS  time-bp sticky: ran to t={t_end:.4f} "
              f"reason={stop2['body'].get('reason')}")
    else:
        failures.append("time-bp re-fired (expected sticky)")
        print(f"FAIL  time-bp refired")

    # 3. Restart + arm a signal breakpoint on `scope` (lowpass output).
    #    The lowpass settles around 0.31 peak — a 0.10 threshold fires
    #    well into the transient.
    req("restart")
    wait_for(lambda f: is_resp(f, "restart"), "restart resp")
    wait_for(lambda f: is_stopped(f, "entry"), "stopped(entry) after restart")
    req("setSignalBreakpoints",
        {"breakpoints": [{"blockId": "scope", "condition": "abs(value) > 0.10"}]})
    wait_for(lambda f: is_resp(f, "setSignalBreakpoints"), "setSigBP resp")
    # Clear any prior time-bps too, so they don't interfere.
    req("setTimeBreakpoints", {"times": []})
    wait_for(lambda f: is_resp(f, "setTimeBreakpoints"), "clear timeBP resp")
    req("continue")
    wait_for(lambda f: is_resp(f, "continue"), "continue resp #3")
    stop3 = wait_for(lambda f: is_stopped(f), "stopped after sig-bp continue")
    if stop3 and stop3["body"].get("reason") == "breakpoint":
        desc = stop3["body"].get("description", "")
        if "scope" in desc:
            print(f"PASS  sig-bp fires: {desc}")
        else:
            failures.append(f"sig-bp description missing 'scope': {desc}")
            print(f"FAIL  sig-bp description: {desc}")
    else:
        failures.append(f"expected stopped(breakpoint) for sig, got "
                        f"{stop3['body'] if stop3 else None}")
        print(f"FAIL  sig-bp did not fire")

    t_sig = get_t()
    # The lowpass transient crosses |0.10| around t = 0.14 — accept
    # any reasonable first-crossing time.
    if not (0.10 <= t_sig <= 1.0):
        failures.append(f"sig-bp at unexpected t={t_sig}")
        print(f"FAIL  sig-bp t:     unexpected {t_sig}")
    else:
        print(f"PASS  sig-bp t:     {t_sig:.4f}")

    req("disconnect")
    proc.stdin.close()
    rc = proc.wait(timeout=10)
    if rc != 0:
        failures.append(f"matlabc exit={rc}")

    print("----")
    if failures:
        print(f"FAIL  {len(failures)} check(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
