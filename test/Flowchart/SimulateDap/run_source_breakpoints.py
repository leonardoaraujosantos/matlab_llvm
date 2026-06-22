#!/usr/bin/env python3
"""#354 — source-line breakpoints inside a MATLAB Function block.

Drives `matlabc -simulate --sim-dap` and verifies that a DAP `setBreakpoints`
on a `signal_matlab_fcn` block's body line pauses the run when the interpreter
reaches that line:

  1. `setBreakpoints { source: { name: <blockId> }, breakpoints: [{ line }] }`
     verifies the line and arms it on the block.
  2. After `continue`, the run stops with `reason: "breakpoint"` and a
     `description` of `<blockId>:<line>`.
  3. A breakpoint on a line that the body never reaches does NOT stop early
     (the run advances past the entry stop to stopTime).

Usage: run_source_breakpoints.py <path-to-matlabc>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def frame(o):
    b = json.dumps(o).encode()
    return f"Content-Length: {len(b)}\r\n\r\n".encode() + b


def read_msg(stream):
    buf = b""
    while b"\r\n\r\n" not in buf:
        c = stream.read(1)
        if not c:
            return None
        buf += c
    clen = 0
    for line in buf.split(b"\r\n"):
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
        here, "..", "..", "..", "examples", "mflowlink",
        "matlab_fcn_breakpoint.mflow"))
    failures = []

    def session(line):
        """Run a session arming `line`; return the first post-continue stop."""
        proc = subprocess.Popen(
            [matlabc, "-simulate", "--sim-dap", mflow],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        seq = [0]

        def send(cmd, args=None):
            seq[0] += 1
            proc.stdin.write(frame({"seq": seq[0], "type": "request",
                                    "command": cmd, "arguments": args or {}}))
            proc.stdin.flush()

        def wait_stop():
            for _ in range(400):
                m = read_msg(proc.stdout)
                if m is None:
                    return None
                if m.get("type") == "event" and m.get("event") == "stopped":
                    return m.get("body", {})
            return None

        def wait_response(cmd):
            for _ in range(400):
                m = read_msg(proc.stdout)
                if m is None:
                    return None
                if m.get("type") == "response" and m.get("command") == cmd:
                    return m
            return None

        send("initialize", {"adapterID": "x"})
        send("launch", {})
        send("setBreakpoints",
             {"source": {"name": "fn"}, "breakpoints": [{"line": line}]})
        bp_resp = wait_response("setBreakpoints")
        send("configurationDone")
        wait_stop()           # the boot/entry stop
        send("continue", {})
        stop = wait_stop()    # the post-continue stop
        # Inspect the body's locals at the stop (the "Locals" scope).
        send("scopes", {"frameId": 1})
        sc = wait_response("scopes")
        scope_names = [s.get("name") for s in sc.get("body", {}).get("scopes", [])] if sc else []
        send("variables", {"variablesReference": 2})
        vr = wait_response("variables")
        locals_ = {x["name"]: x["value"]
                   for x in (vr.get("body", {}).get("variables", []) if vr else [])}
        proc.kill()
        return bp_resp, stop, scope_names, locals_

    # Case 1 — breakpoint on line 3 ("b = a * 2;") pauses the run; the Locals
    # scope shows the body's variables *before* that line runs (a = u1+1 = 4,
    # u1 = 3), and `b` is not yet assigned.
    bp_resp, stop, scopes, locals_ = session(3)
    verified = False
    if bp_resp:
        bps = bp_resp.get("body", {}).get("breakpoints", [])
        verified = bool(bps) and bps[0].get("verified") is True
    if not verified:
        failures.append("setBreakpoints did not verify the armed line")
    if not stop or stop.get("reason") != "breakpoint":
        failures.append(f"continue did not stop on the breakpoint (got {stop})")
    elif stop.get("description") != "fn:3":
        failures.append(
            f"stopped description should be 'fn:3', got {stop.get('description')}")
    if "Locals" not in scopes:
        failures.append(f"no Locals scope at the breakpoint (scopes: {scopes})")
    if abs(float(locals_.get("a", "nan")) - 4.0) > 1e-9:
        failures.append(f"local 'a' should be 4 at fn:3, got {locals_.get('a')}")
    if abs(float(locals_.get("u1", "nan")) - 3.0) > 1e-9:
        failures.append(f"local 'u1' should be 3 at fn:3, got {locals_.get('u1')}")
    if "b" in locals_:
        failures.append("local 'b' must not exist yet at fn:3 (line not run)")

    # Case 2 — a breakpoint on a non-existent body line (99) never fires; the
    # run reaches stopTime (reason != breakpoint).
    _, stop2, _, _ = session(99)
    if stop2 and stop2.get("reason") == "breakpoint":
        failures.append("a breakpoint on an unreached line must not stop the run")

    if failures:
        print("FAIL: source-line breakpoints inside MATLAB Function blocks")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: breakpoint stops at fn:3 with Locals (a=4, u1=3, no b); "
          "unreached line does not stop")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: run_source_breakpoints.py <path-to-matlabc>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
