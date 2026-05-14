#!/usr/bin/env python3
"""Tier-D smoke test for `matlabc -simulate --sim-dap`.

Drives the DAP server on stdio with a fixed scripted conversation and
verifies the simulator pauses at entry, advances on `next`, restores
on `stepBack`, and resumes to stopTime on `continue`.

Usage: run_tests.py <path-to-matlabc>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def frame(obj: dict) -> bytes:
    body = json.dumps(obj).encode()
    return f"Content-Length: {len(body)}\r\n\r\n".encode() + body


def read_one_frame(stream):
    """Read exactly one DAP frame from a binary stream. Returns the
    parsed JSON dict, or None on EOF."""
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


def main(matlabc: str) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    mflow = os.path.abspath(
        os.path.join(here, "..", "..", "..", "examples",
                     "mflowlink", "lowpass.mflow"))

    proc = subprocess.Popen(
        [matlabc, "-simulate", "--sim-dap", mflow],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout

    seq = [10]
    failures: list[str] = []

    def send(obj):
        proc.stdin.write(frame(obj))
        proc.stdin.flush()

    def req(cmd, args=None):
        seq[0] += 1
        send({"seq": seq[0], "type": "request", "command": cmd,
              "arguments": args or {}})

    def fail(msg: str):
        failures.append(msg)
        print(f"FAIL  {msg}")

    def wait_for(pred, label, max_frames=400):
        for _ in range(max_frames):
            f = read_one_frame(proc.stdout)
            if f is None:
                fail(f"stream closed while waiting for {label}")
                return None
            if pred(f):
                return f
        fail(f"timed out waiting for {label}")
        return None

    def is_stopped(f, reason=None):
        return (f.get("type") == "event" and f.get("event") == "stopped"
                and (reason is None or f.get("body", {}).get("reason") == reason))

    def is_resp(f, cmd):
        return f.get("type") == "response" and f.get("command") == cmd

    # 1. initialize → response + initialized event.
    req("initialize")
    wait_for(lambda f: is_resp(f, "initialize"), "initialize response")
    wait_for(lambda f: f.get("event") == "initialized", "initialized event")

    # 2. launch + configurationDone → stopped(entry).
    req("launch")
    wait_for(lambda f: is_resp(f, "launch"), "launch response")
    req("configurationDone")
    wait_for(lambda f: is_resp(f, "configurationDone"), "confDone response")
    entry = wait_for(lambda f: is_stopped(f, "entry"), "stopped(entry)")
    if entry:
        print(f"PASS  entry stop:  {entry['body'].get('reason')}")

    def get_t() -> float:
        req("variables", {"variablesReference": 1})
        resp = wait_for(lambda f: is_resp(f, "variables"), "variables resp")
        if not resp:
            return float("nan")
        for v in resp["body"]["variables"]:
            if v["name"] == "t":
                return float(v["value"])
        fail("no t in variables")
        return float("nan")

    t0 = get_t()
    if t0 != 0.0:
        fail(f"expected t=0 at entry, got {t0}")
    else:
        print(f"PASS  entry t=0:   {t0}")

    # 3. next × 5 → t advances by 5 · 0.01 = 0.05.
    for _ in range(5):
        req("next")
        wait_for(lambda f: is_resp(f, "next"), "next resp")
        wait_for(lambda f: is_stopped(f, "step"), "stopped(step)")
    t5 = get_t()
    if not (0.045 <= t5 <= 0.055):
        fail(f"expected t≈0.05 after 5 steps, got {t5}")
    else:
        print(f"PASS  5 steps:     t={t5:.4f}")

    # 4. stepBack × 3 → t back to ~0.02.
    for _ in range(3):
        req("stepBack")
        wait_for(lambda f: is_resp(f, "stepBack"), "stepBack resp")
        wait_for(lambda f: is_stopped(f, "step"), "stopped(step-back)")
    t_back = get_t()
    if not (0.015 <= t_back <= 0.025):
        fail(f"expected t≈0.02 after 3 step-backs, got {t_back}")
    else:
        print(f"PASS  3 stepBack:  t={t_back:.4f}")

    # 5. continue → t reaches stopTime = 10.
    req("continue")
    wait_for(lambda f: is_resp(f, "continue"), "continue resp")
    wait_for(lambda f: is_stopped(f), "stopped after continue")
    t_end = get_t()
    if not (9.99 <= t_end <= 10.001):
        fail(f"expected t≈10.0 after continue, got {t_end}")
    else:
        print(f"PASS  continue:    t={t_end:.4f}")

    # 6. disconnect → clean exit.
    req("disconnect")
    proc.stdin.close()
    rc = proc.wait(timeout=10)
    if rc != 0:
        fail(f"matlabc exited with {rc}")
    else:
        print(f"PASS  disconnect:  exit {rc}")

    print("----")
    if failures:
        print(f"FAIL  {len(failures)} check(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <matlabc>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
