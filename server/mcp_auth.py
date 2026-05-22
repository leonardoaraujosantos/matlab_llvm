"""Backend-minted MCP tokens (stateless, HMAC-signed).

An authenticated REST caller mints a token (``POST /v1/mcp/token``) bound to
their identity; the MCP client presents it as a bearer on ``/mcp``. Tokens are
verified locally (no DB, no per-request call to CyberdyneAuth):

    <base64url(payload)>.<base64url(hmac_sha256(secret, payload))>

The payload carries ``sub`` (identity id), ``email``, ``typ=mcp``, ``iat``,
``exp``. Revocation is by expiry or secret rotation (a denylist arrives with
the DB layer — see docs/lms_roadmap.md).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from config import settings


def _secret() -> bytes:
    key = settings.mcp_token_secret or settings.api_token or "dev-insecure-mcp-secret"
    return key.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body: str) -> str:
    return _b64e(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())


def mint(subject: str, email: str | None = None, ttl_s: int | None = None) -> tuple[str, int]:
    """Return (token, exp_epoch) for ``subject``."""
    now = int(time.time())
    exp = now + (ttl_s if ttl_s is not None else settings.mcp_token_ttl_s)
    payload = {"sub": subject, "email": email, "typ": "mcp", "iat": now, "exp": exp}
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{body}.{_sign(body)}", exp


def verify(token: str) -> dict | None:
    """Return the payload if the token is well-formed, untampered, and unexpired."""
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_b64d(body))
    except Exception:
        return None
    if payload.get("typ") != "mcp":
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload
