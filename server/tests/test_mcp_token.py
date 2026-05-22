"""MCP token minting + verification."""

import asyncio

import auth
import mcp_auth


# --- token mint / verify --------------------------------------------------
def test_mint_verify_roundtrip(monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "test-secret")
    token, exp = mcp_auth.mint("u-123", "a@b.co")
    payload = mcp_auth.verify(token)
    assert payload is not None
    assert payload["sub"] == "u-123"
    assert payload["email"] == "a@b.co"
    assert payload["typ"] == "mcp"
    assert payload["exp"] == exp


def test_verify_rejects_tamper_and_garbage(monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "test-secret")
    token, _ = mcp_auth.mint("u-1")
    body, sig = token.split(".", 1)
    assert mcp_auth.verify(f"{body}.{sig[:-2]}xx") is None  # bad signature
    assert mcp_auth.verify("garbage") is None
    assert mcp_auth.verify("a.b.c") is None


def test_verify_rejects_expired(monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "test-secret")
    token, _ = mcp_auth.mint("u-1", ttl_s=-10)
    assert mcp_auth.verify(token) is None


def test_verify_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "secret-a")
    token, _ = mcp_auth.mint("u-1")
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "secret-b")
    assert mcp_auth.verify(token) is None


def test_verifier_maps_token_to_access_token(monkeypatch):
    from mcp_tools import McpTokenVerifier

    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "test-secret")
    token, _ = mcp_auth.mint("u-xyz")
    v = McpTokenVerifier()
    at = asyncio.run(v.verify_token(token))
    assert at is not None and at.client_id == "u-xyz"
    assert asyncio.run(v.verify_token("bad.token")) is None


# --- POST /v1/mcp/token ---------------------------------------------------
class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
        self.text = ""

    def json(self):
        return self._p


class _FakeAuthClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, **k):
        tok = (headers or {}).get("Authorization", "").removeprefix("Bearer ").strip()
        if tok == "alice":
            return _Resp(200, {"id": "u-alice", "email": "alice@x.io", "is_active": True})
        return _Resp(401, {})


def test_mint_endpoint_open_mode_binds_anon(client, monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "ts")
    r = client.post("/v1/mcp/token")
    assert r.status_code == 200
    body = r.json()
    assert body["subject"] == "anon"
    assert body["token_type"] == "bearer"
    assert mcp_auth.verify(body["token"])["sub"] == "anon"


def test_mint_endpoint_binds_identity_and_requires_auth(client, monkeypatch):
    monkeypatch.setattr(mcp_auth.settings, "mcp_token_secret", "ts")
    monkeypatch.setattr(auth.settings, "cyberdyne_auth_url", "http://auth.test")
    monkeypatch.setattr(auth.httpx, "AsyncClient", _FakeAuthClient)
    auth._cache.clear()
    r = client.post("/v1/mcp/token", headers={"Authorization": "Bearer alice"})
    assert r.status_code == 200
    assert r.json()["subject"] == "u-alice"
    # unauthenticated minting is rejected in cyberdyne mode
    assert client.post("/v1/mcp/token").status_code == 401
