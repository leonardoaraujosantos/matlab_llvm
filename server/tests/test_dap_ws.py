"""DAP-over-WebSocket bridge tests.

Drives the bridge against the fake matlabc's ``-dap`` server (conftest.py),
so it validates the byte pumping / framing passthrough / lifecycle without a
real compiler build. The bridge is an opaque byte stream, so the client
reassembles DAP Content-Length frames itself.
"""

import io
import json


def _put_program(client, session, program="program.m", body=b"x = 42;\ndisp(x)\n"):
    r = client.post(
        "/v1/files",
        params={"session_id": session},
        files={"file": (program, io.BytesIO(body), "text/plain")},
    )
    assert r.status_code == 200, r.text


class _Frames:
    """Reassemble DAP frames out of the WS byte stream (skips text diags)."""

    def __init__(self, ws):
        self.ws = ws
        self.buf = b""

    def _fill(self):
        while True:
            msg = self.ws.receive()
            if msg.get("type") == "websocket.close":
                raise EOFError("ws closed")
            if msg.get("bytes") is not None:
                self.buf += msg["bytes"]
                return
            # text message = stderr diagnostic; ignore and keep reading

    def read(self):
        while True:
            idx = self.buf.find(b"\r\n\r\n")
            if idx != -1:
                length = 0
                for ln in self.buf[:idx].split(b"\r\n"):
                    if ln.lower().startswith(b"content-length:"):
                        length = int(ln.split(b":", 1)[1])
                if len(self.buf) >= idx + 4 + length:
                    body = self.buf[idx + 4 : idx + 4 + length]
                    self.buf = self.buf[idx + 4 + length :]
                    return json.loads(body.decode("utf-8"))
            self._fill()


def _send(ws, seq, command, arguments=None):
    body = json.dumps(
        {"seq": seq, "type": "request", "command": command, "arguments": arguments or {}}
    ).encode("utf-8")
    ws.send_bytes(b"Content-Length: %d\r\n\r\n" % len(body) + body)


def test_dap_ws_breakpoint_step_flow(client):
    _put_program(client, "dapsess")
    with client.websocket_connect("/v1/dap/ws/dapsess?program=program.m") as ws:
        frames = _Frames(ws)

        _send(ws, 1, "initialize", {"adapterID": "matlab"})
        first_two = [frames.read(), frames.read()]
        assert any(
            m.get("type") == "response" and m.get("command") == "initialize" and m.get("success")
            for m in first_two
        )
        assert any(m.get("type") == "event" and m.get("event") == "initialized" for m in first_two)

        _send(ws, 2, "launch", {"program": "program.m"})
        stopped = None
        for _ in range(4):
            m = frames.read()
            if m.get("type") == "event" and m.get("event") == "stopped":
                stopped = m
                break
        assert stopped is not None
        assert stopped["body"]["reason"] == "breakpoint"

        # Inspect: stackTrace -> scopes -> variables round-trips through bridge.
        _send(ws, 3, "stackTrace", {"threadId": 1})
        st = _read_response(frames, "stackTrace")
        assert st["body"]["stackFrames"][0]["name"] == "main"

        _send(ws, 4, "variables", {"variablesReference": 1000})
        var = _read_response(frames, "variables")
        assert any(v["name"] == "x" for v in var["body"]["variables"])

        # Continue -> terminated event, then the bridge closes the socket.
        _send(ws, 5, "continue", {"threadId": 1})
        terminated = False
        try:
            for _ in range(6):
                m = frames.read()
                if m.get("type") == "event" and m.get("event") == "terminated":
                    terminated = True
                    break
        except EOFError:
            pass
        assert terminated


def _read_response(frames, command):
    for _ in range(6):
        m = frames.read()
        if m.get("type") == "response" and m.get("command") == command:
            return m
    raise AssertionError(f"no response for {command}")


def test_dap_ws_missing_program_is_rejected(client):
    with client.websocket_connect("/v1/dap/ws/nosuch?program=absent.m") as ws:
        msg = ws.receive()
        assert msg["type"] == "websocket.close"
        assert msg.get("code") == 1008
