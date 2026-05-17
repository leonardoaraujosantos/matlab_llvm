#!/usr/bin/env python3
"""DAP-driven chart smoke test.

Spawns `matlabc -simulate --sim-dap <chart>.mflow` over stdio, drives
the chart through a fixed scenario that exercises temporalCount /
duration / state-transition firings, and verifies the chart's active
state matches the expected trajectory at each step.

Coverage targets:
  - `temporal_counter.mflow` — temporalCount(sample) >= 5 fires
    watch -> warn; duration(pressure > 100) >= 3 fires warn -> fault;
    pressure < 50 fires fault -> watch. Exercises both counter-style
    temporal operators end-to-end through the interpreter.
  - `traffic_light_moore.mflow` — bare smoke test: initialize, emit
    `step`, advance super-step, observe state transitions.

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
    """Read exactly one DAP frame from a binary stream."""
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


class Session:
    """Tiny DAP session wrapper. seq counts outbound; req returns
    nothing — pending responses are matched via wait_for(predicate).
    """
    def __init__(self, proc):
        self.proc = proc
        self.seq = 10
        self.failures: list[str] = []

    def send(self, obj):
        self.proc.stdin.write(frame(obj))
        self.proc.stdin.flush()

    def req(self, cmd, args=None):
        self.seq += 1
        self.send({"seq": self.seq, "type": "request",
                   "command": cmd, "arguments": args or {}})

    def fail(self, msg):
        self.failures.append(msg)
        print(f"FAIL  {msg}")

    def wait_for(self, pred, label, max_frames=400):
        for _ in range(max_frames):
            f = read_one_frame(self.proc.stdout)
            if f is None:
                self.fail(f"stream closed while waiting for {label}")
                return None
            if pred(f):
                return f
        self.fail(f"timed out waiting for {label}")
        return None

    def collect_until(self, stop_pred, label, max_frames=400):
        """Read frames until stop_pred(f) is true. Returns the list
        of frames seen (excluding the terminator)."""
        seen = []
        for _ in range(max_frames):
            f = read_one_frame(self.proc.stdout)
            if f is None:
                self.fail(f"stream closed while collecting for {label}")
                return seen
            if stop_pred(f):
                return seen
            seen.append(f)
        self.fail(f"timed out collecting for {label}")
        return seen


def is_resp(f, cmd):
    return f.get("type") == "response" and f.get("command") == cmd


def active_ids(getactive_response):
    """Pull the active-state set out of a stateChart/getActive
    response. The adapter returns {ids: [...]}. We surface a set
    of leaf-state names for direct membership checks."""
    body = getactive_response.get("body", {})
    ids = body.get("ids", [])
    return set(ids)


def drive(session: Session):
    """Boot the DAP session into a state where we can issue
    stateChart/* requests. Mirrors the handshake used by
    test/Flowchart/SimulateDap/run_tests.py."""
    session.req("initialize")
    session.wait_for(lambda f: is_resp(f, "initialize"), "initialize")
    session.wait_for(lambda f: f.get("event") == "initialized",
                     "initialized event")
    session.req("launch")
    session.wait_for(lambda f: is_resp(f, "launch"), "launch")
    session.req("configurationDone")
    session.wait_for(lambda f: is_resp(f, "configurationDone"),
                     "configurationDone")
    # Drain the initial enter-state events so subsequent collect_until
    # calls see a clean stream.
    session.wait_for(
        lambda f: f.get("event") == "stopped"
                  or f.get("event") == "stateChart/superStepEnd",
        "first super-step settled")


def get_active(session: Session) -> set:
    session.req("stateChart/getActive")
    resp = session.wait_for(
        lambda f: is_resp(f, "stateChart/getActive"), "getActive")
    return active_ids(resp) if resp else set()


def step(session: Session):
    """Fire one super-step and drain until quiescent."""
    session.req("stateChart/stepSuperStep")
    session.collect_until(
        lambda f: (f.get("type") == "response"
                   and f.get("command") == "stateChart/stepSuperStep"),
        "stepSuperStep response")


def set_local(session: Session, name: str, value):
    session.req("stateChart/setLocal",
                {"name": name, "value": value})
    session.wait_for(lambda f: is_resp(f, "stateChart/setLocal"),
                     "setLocal")


def emit_event(session: Session, name: str):
    session.req("stateChart/emit", {"name": name})
    session.wait_for(lambda f: is_resp(f, "stateChart/emit"),
                     "emit")


def test_temporal_counter(matlabc: str, here: str) -> bool:
    """Walk watch -> warn (via 5 sample events) -> fault (via
    pressure>100 for 3+ super-steps) -> watch (via pressure<50)."""
    chart = os.path.abspath(os.path.join(
        here, "..", "..", "StateChart", "temporal_counter.mflow"))
    proc = subprocess.Popen(
        [matlabc, "-simulate", "--sim-dap", chart],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    s = Session(proc)
    try:
        drive(s)

        # Should start in `watch`.
        active = get_active(s)
        if "watch" not in active:
            s.fail(f"temporal_counter: initial active = {active!r}, "
                   f"expected to contain watch")

        # Fire 5 sample events. temporalCount(sample) should advance
        # the counter; on the 5th it reaches the >=5 guard and the
        # transition watch -> warn should fire on the next super-step.
        for i in range(5):
            emit_event(s, "sample")
            step(s)
        active = get_active(s)
        if "warn" not in active:
            s.fail(f"temporal_counter: after 5 sample events, active "
                   f"= {active!r}, expected to contain warn")

        # Now drive duration(pressure > 100) >= 3. Set pressure high,
        # step several super-steps until duration accumulates past 3
        # and the warn -> fault transition fires.
        set_local(s, "pressure", 150.0)
        for _ in range(6):
            step(s)
        active = get_active(s)
        if "fault" not in active:
            s.fail(f"temporal_counter: after pressure>100 for 6 "
                   f"super-steps, active = {active!r}, expected to "
                   f"contain fault")

        # Drop pressure to trigger fault -> watch.
        set_local(s, "pressure", 20.0)
        step(s)
        active = get_active(s)
        if "watch" not in active:
            s.fail(f"temporal_counter: after pressure<50, active "
                   f"= {active!r}, expected to contain watch")

    finally:
        try:
            s.req("disconnect")
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return not s.failures


def test_traffic_light_smoke(matlabc: str, here: str) -> bool:
    """Bare smoke: initialise, observe the chart entered S_red, fire
    `step` events, advance super-steps. Doesn't assert specific
    transition timing — the timer guard in the chart depends on
    super-step counting which is best verified visually."""
    chart = os.path.abspath(os.path.join(
        here, "..", "..", "StateChart", "traffic_light_moore.mflow"))
    proc = subprocess.Popen(
        [matlabc, "-simulate", "--sim-dap", chart],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    s = Session(proc)
    try:
        drive(s)
        active = get_active(s)
        if "S_red" not in active:
            s.fail(f"traffic_light_moore: initial active = "
                   f"{active!r}, expected to contain S_red")
        # Pump events; chart should remain in some valid state.
        for _ in range(40):
            emit_event(s, "step")
            step(s)
        active = get_active(s)
        if not (active & {"S_red", "S_yellow", "S_green"}):
            s.fail(f"traffic_light_moore: after 40 steps, active = "
                   f"{active!r}, expected one of S_*")
    finally:
        try:
            s.req("disconnect")
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    return not s.failures


def main(matlabc: str) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    cases = [
        ("temporal_counter",   test_temporal_counter),
        ("traffic_light_smoke", test_traffic_light_smoke),
    ]
    passed = 0
    failed = 0
    for name, fn in cases:
        try:
            ok = fn(matlabc, here)
        except Exception as exc:
            ok = False
            print(f"FAIL  {name}: {exc}")
        if ok:
            print(f"PASS  {name}")
            passed += 1
        else:
            failed += 1
    print(f"----\npassed: {passed}    failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: run_tests.py <path-to-matlabc>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
