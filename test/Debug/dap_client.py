"""Minimal DAP client for testing matlabc's -dap mode.

Spawns matlabc as a subprocess and speaks the LSP-style framed JSON
protocol over its stdin/stdout. A reader thread demuxes incoming
frames into two queues — responses (keyed by request_seq) and events
(keyed by event name). Test scenarios use `request(cmd, args)` to send
a request and block on its response, and `wait_event(name)` to block
until a particular event arrives.

Everything is bounded by short timeouts so a hung server fails the
test instead of hanging the suite. The client takes the matlabc path
and the program path as constructor arguments — there's no fallback
discovery.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time


class DapError(RuntimeError):
    pass


class DapClient:
    def __init__(self, matlabc, program, trace=False):
        self.matlabc = matlabc
        self.program = program
        env = os.environ.copy()
        if trace:
            env["MATLABC_DAP_TRACE"] = "1"
        self.proc = subprocess.Popen(
            [matlabc, "-dap", program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._next_seq = 1
        self._lock = threading.Lock()
        self._responses = {}
        self._response_cv = threading.Condition()
        # Each event name has its own queue. wait_event drains in FIFO order.
        self._event_queues = {}
        self._event_cv = threading.Condition()
        self._reader_done = threading.Event()
        self._reader_err = None
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Surface stderr to the test driver if anything goes wrong.
        self._stderr_buf = []
        self._stderr_reader = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()

    # --- subprocess plumbing -------------------------------------------------

    def _drain_stderr(self):
        try:
            for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
                if not chunk:
                    break
                self._stderr_buf.append(chunk.decode("utf-8", "replace"))
        except Exception:
            pass

    def _read_frame(self):
        # Read headers up to blank line, then exactly Content-Length bytes.
        headers = b""
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                return None
            headers += ch
            if headers.endswith(b"\r\n\r\n"):
                break
            if len(headers) > 64 * 1024:
                raise DapError("oversized header block from server")
        length = None
        for line in headers.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1].strip())
                break
        if length is None:
            raise DapError(f"frame missing Content-Length: {headers!r}")
        body = b""
        while len(body) < length:
            chunk = self.proc.stdout.read(length - len(body))
            if not chunk:
                return None
            body += chunk
        return json.loads(body.decode("utf-8"))

    def _read_loop(self):
        try:
            while True:
                msg = self._read_frame()
                if msg is None:
                    return
                kind = msg.get("type")
                if kind == "response":
                    rs = msg.get("request_seq")
                    with self._response_cv:
                        self._responses[rs] = msg
                        self._response_cv.notify_all()
                elif kind == "event":
                    name = msg.get("event")
                    with self._event_cv:
                        self._event_queues.setdefault(
                            name, queue.Queue()).put(msg)
                        self._event_cv.notify_all()
        except Exception as e:
            self._reader_err = e
        finally:
            self._reader_done.set()
            with self._response_cv:
                self._response_cv.notify_all()
            with self._event_cv:
                self._event_cv.notify_all()

    # --- request / event API -------------------------------------------------

    def request(self, command, arguments=None, timeout=30.0):
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            msg = {
                "seq": seq,
                "type": "request",
                "command": command,
                "arguments": arguments or {},
            }
            body = json.dumps(msg).encode("utf-8")
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            try:
                self.proc.stdin.write(header)
                self.proc.stdin.write(body)
                self.proc.stdin.flush()
            except BrokenPipeError as e:
                raise DapError(
                    f"stdin closed while sending {command}: {e}") from e
        deadline = time.monotonic() + timeout
        with self._response_cv:
            while seq not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DapError(
                        f"timeout waiting for response to {command} "
                        f"(seq={seq})")
                if self._reader_done.is_set():
                    raise DapError(
                        f"reader stopped before {command} response: "
                        f"{self._reader_err}; stderr={''.join(self._stderr_buf)}")
                self._response_cv.wait(timeout=remaining)
            resp = self._responses.pop(seq)
        if not resp.get("success", False):
            raise DapError(
                f"{command} failed: {resp.get('message')!r}; full={resp}")
        return resp.get("body", {}) or {}

    def wait_event(self, name, timeout=30.0, predicate=None):
        """Block until an event with the given name arrives.

        If `predicate` is given, drains events from the queue until one
        satisfies it. Useful for filtering 'output' events to a specific
        category, etc."""
        deadline = time.monotonic() + timeout
        with self._event_cv:
            while True:
                q = self._event_queues.setdefault(name, queue.Queue())
                while not q.empty():
                    msg = q.get_nowait()
                    if predicate is None or predicate(msg):
                        return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DapError(
                        f"timeout waiting for event {name!r}; "
                        f"stderr={''.join(self._stderr_buf)}")
                if self._reader_done.is_set():
                    raise DapError(
                        f"reader stopped while waiting for {name!r}: "
                        f"{self._reader_err}; "
                        f"stderr={''.join(self._stderr_buf)}")
                self._event_cv.wait(timeout=remaining)

    def expect_no_event(self, name, window=0.4):
        """Assert no event of `name` arrives within `window` seconds.

        Used for log-point scenarios where we want to confirm the
        server did NOT emit a 'stopped' event for the silently-resumed
        breakpoint."""
        deadline = time.monotonic() + window
        with self._event_cv:
            while True:
                q = self._event_queues.setdefault(name, queue.Queue())
                if not q.empty():
                    raise DapError(
                        f"unexpected event {name!r}: {q.get_nowait()!r}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._event_cv.wait(timeout=remaining)

    # --- lifecycle -----------------------------------------------------------

    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    self.request("disconnect", timeout=2.0)
                except Exception:
                    pass
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --- standard launch helpers -------------------------------------------------

def initialize_and_launch(client, stop_on_entry=False, breakpoints=None):
    """Run the standard DAP boot sequence:
       initialize -> wait initialized -> launch -> setBreakpoints
       -> configurationDone. Returns once the worker has been started."""
    caps = client.request("initialize", {
        "clientID": "matlabc-test",
        "linesStartAt1": True,
        "columnsStartAt1": True,
        "pathFormat": "path",
    })
    assert caps.get("supportsConfigurationDoneRequest"), \
        f"server missing supportsConfigurationDoneRequest cap: {caps}"
    client.wait_event("initialized", timeout=5.0)
    client.request("launch", {
        "program": client.program,
        "stopOnEntry": stop_on_entry,
    })
    if breakpoints:
        body = client.request("setBreakpoints", {
            "source": {"path": client.program},
            "breakpoints": breakpoints,
        })
        verified = body.get("breakpoints") or []
        for bp in verified:
            if not bp.get("verified"):
                raise DapError(
                    f"breakpoint not verified: {bp}; full={body}")
    client.request("configurationDone")
