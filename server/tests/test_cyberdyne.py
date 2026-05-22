"""CyberdyneAuth integration tests (mocked /api/v1/users/me via httpx)."""

import io
import json

import auth

_CALLS = {"n": 0}


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
        self.text = ""

    def json(self):
        return self._p


class _FakeAuthClient:
    """Maps the bearer token to a /users/me response."""

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, **k):
        _CALLS["n"] += 1
        tok = (headers or {}).get("Authorization", "").removeprefix("Bearer ").strip()
        table = {
            "good": (200, {"id": "u-123", "email": "a@b.co", "is_active": True}),
            "alice": (200, {"id": "u-alice", "is_active": True}),
            "bob": (200, {"id": "u-bob", "is_active": True}),
            "boom": (500, {}),
        }
        status, payload = table.get(tok, (401, {}))
        return _Resp(status, payload)


def _enable(monkeypatch):
    monkeypatch.setattr(auth.settings, "cyberdyne_auth_url", "http://auth.test")
    monkeypatch.setattr(auth.httpx, "AsyncClient", _FakeAuthClient)
    auth._cache.clear()
    _CALLS["n"] = 0


def test_valid_token_allows_and_whoami_reports_identity(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/v1/check", json={"source": "x = 1;"}, headers={"Authorization": "Bearer good"})
    assert r.status_code == 200
    w = client.get("/v1/auth/whoami", headers={"Authorization": "Bearer good"}).json()
    assert w["authenticated"] is True
    assert w["mode"] == "cyberdyne"
    assert w["id"] == "u-123"


def test_invalid_token_rejected(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/v1/check", json={"source": "x = 1;"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_missing_token_rejected(client, monkeypatch):
    _enable(monkeypatch)
    assert client.post("/v1/check", json={"source": "x = 1;"}).status_code == 401


def test_auth_service_down_returns_503(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/v1/check", json={"source": "x = 1;"}, headers={"Authorization": "Bearer boom"})
    assert r.status_code == 503


def test_verification_is_cached(client, monkeypatch):
    _enable(monkeypatch)
    for _ in range(3):
        client.post("/v1/check", json={"source": "x = 1;"}, headers={"Authorization": "Bearer good"})
    assert _CALLS["n"] == 1  # only the first call hits the auth service


def test_workspace_isolated_per_identity(client, monkeypatch):
    _enable(monkeypatch)
    # alice writes a file to session "shared"
    client.post(
        "/v1/files",
        params={"session_id": "shared"},
        files={"file": ("a.txt", io.BytesIO(b"secret"), "text/plain")},
        headers={"Authorization": "Bearer alice"},
    )
    # bob, hitting the same session_id, must NOT see alice's file
    bob = client.get(
        "/v1/files", params={"session_id": "shared"}, headers={"Authorization": "Bearer bob"}
    ).json()
    assert all(f["path"] != "a.txt" for f in bob["files"])
    # alice still sees her own
    alice = client.get(
        "/v1/files", params={"session_id": "shared"}, headers={"Authorization": "Bearer alice"}
    ).json()
    assert any(f["path"] == "a.txt" for f in alice["files"])


def test_whoami_open_mode(client):
    w = client.get("/v1/auth/whoami").json()
    assert w["authenticated"] is False
    assert w["mode"] == "none"


def test_dap_ws_rejects_bad_token(client, monkeypatch):
    _enable(monkeypatch)
    client.post(
        "/v1/files",
        params={"session_id": "dws"},
        files={"file": ("program.m", io.BytesIO(b"x = 1;\n"), "text/plain")},
        headers={"Authorization": "Bearer alice"},
    )
    with client.websocket_connect("/v1/dap/ws/dws?program=program.m&token=nope") as ws:
        msg = ws.receive()
        assert msg["type"] == "websocket.close"
        assert msg.get("code") == 1008


def test_dap_ws_accepts_valid_token(client, monkeypatch):
    _enable(monkeypatch)
    client.post(
        "/v1/files",
        params={"session_id": "dws2"},
        files={"file": ("program.m", io.BytesIO(b"x = 1;\n"), "text/plain")},
        headers={"Authorization": "Bearer alice"},
    )
    with client.websocket_connect("/v1/dap/ws/dws2?program=program.m&token=alice") as ws:
        req = {"seq": 1, "type": "request", "command": "initialize", "arguments": {}}
        body = json.dumps(req).encode("utf-8")
        ws.send_bytes(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        msg = ws.receive()
        assert msg["type"] == "websocket.send"
        assert msg.get("bytes") is not None  # a DAP frame, not an auth close
