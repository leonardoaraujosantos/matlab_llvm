#!/usr/bin/env python3
"""Tier-E block-stepping smoke test for `matlabc -simulate --sim-dap`.

Drives the DAP server through one major step of `lowpass.mflow` by
firing `stepBlock` four times (one per block in the topo order) and
checking that the simulator emits the expected `simulationActiveBlock`
sequence: src → k → lp → scope. A fifth `stepBlock` should commit a
major step (BlockCursor wraps).

Usage: run_block_step.py <path-to-matlabc>
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

    def wait_for(pred, label, max_frames=300):
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

    def is_stopped(f):
        return f.get("type") == "event" and f.get("event") == "stopped"

    def is_active_block(f):
        return (f.get("type") == "event" and
                f.get("event") == "simulationActiveBlock")

    # Handshake.
    req("initialize")
    wait_for(lambda f: is_resp(f, "initialize"), "init resp")
    wait_for(lambda f: f.get("event") == "initialized", "initialized")
    req("launch")
    wait_for(lambda f: is_resp(f, "launch"), "launch resp")
    req("configurationDone")
    wait_for(lambda f: is_resp(f, "configurationDone"), "confDone resp")
    wait_for(lambda f: is_stopped(f) and f["body"].get("reason") == "entry",
             "stopped(entry)")

    expected = ["src", "k", "lp", "scope"]
    seen = []
    # Fire stepBlock 4 times — one per block.
    for label in expected:
        req("stepBlock")
        wait_for(lambda f: is_resp(f, "stepBlock"), f"stepBlock resp ({label})")
        ev = wait_for(lambda f: is_active_block(f),
                      f"active block ({label})")
        if ev:
            seen.append(ev["body"].get("nodeId"))
        wait_for(lambda f: is_stopped(f), f"stopped after stepBlock ({label})")

    if seen == expected:
        print(f"PASS  block order: {seen}")
    else:
        failures.append(f"expected {expected}, got {seen}")
        print(f"FAIL  block order: got {seen}, expected {expected}")

    # A fifth stepBlock should wrap and commit a major step. We can
    # observe this by checking that majorSteps incremented (via
    # `variables` t > 0).
    def get_t():
        req("variables", {"variablesReference": 1})
        r = wait_for(lambda f: is_resp(f, "variables"), "variables")
        if not r:
            return float("nan")
        for v in r["body"]["variables"]:
            if v["name"] == "t":
                return float(v["value"])
        return float("nan")

    t_before = get_t()
    req("stepBlock")
    wait_for(lambda f: is_resp(f, "stepBlock"), "stepBlock resp (commit)")
    wait_for(lambda f: is_stopped(f), "stopped after commit")
    t_after = get_t()
    if t_after > t_before + 1e-6:
        print(f"PASS  major-step commit: t {t_before:.4f} -> {t_after:.4f}")
    else:
        failures.append(f"5th stepBlock did not commit major step "
                        f"({t_before} → {t_after})")
        print(f"FAIL  major-step commit: t did not advance")

    # stepBackBlock from cursor-zero (just after a major step commit)
    # should pop the previous major step.
    req("stepBackBlock")
    wait_for(lambda f: is_resp(f, "stepBackBlock"), "stepBackBlock resp")
    wait_for(lambda f: is_stopped(f), "stopped after stepBackBlock")
    t_back = get_t()
    if abs(t_back - t_before) < 1e-6:
        print(f"PASS  stepBackBlock@0: t -> {t_back:.4f}")
    else:
        failures.append(f"stepBackBlock did not restore prior major step "
                        f"({t_after} → {t_back}, want {t_before})")
        print(f"FAIL  stepBackBlock@0")

    req("disconnect")
    proc.stdin.close()
    rc = proc.wait(timeout=5)
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
