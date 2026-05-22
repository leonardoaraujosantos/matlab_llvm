"""Authentication.

Three modes, chosen by ``settings.auth_mode``:

* ``cyberdyne`` — validate the caller's bearer against CyberdyneAuth's
  ``GET /api/v1/users/me`` (200 = valid + profile; 401/403 = reject; 5xx =
  service down). Successful checks are cached briefly to avoid hammering the
  auth service. The verified identity id becomes the request *principal*, so
  workspaces are isolated per user.
* ``token`` — a single shared static bearer token.
* ``none`` — open (local-dev default).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from fastapi import Header, HTTPException, Request, status

from config import settings
from principal import set_principal

USERS_ME_PATH = "/api/v1/users/me"


@dataclass
class Identity:
    id: str
    email: str | None = None
    organization_id: str | None = None
    is_active: bool = True


# token -> (expiry_monotonic, Identity)
_cache: dict[str, tuple[float, Identity]] = {}


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def verify_cyberdyne(token: str) -> Identity:
    now = time.monotonic()
    cached = _cache.get(token)
    if cached and cached[0] > now:
        return cached[1]

    url = settings.cyberdyne_auth_url.rstrip("/") + USERS_ME_PATH
    try:
        async with httpx.AsyncClient(timeout=settings.cyberdyne_timeout_s) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"auth service unavailable: {exc}")

    if resp.status_code in (401, 403):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        )
    if resp.status_code >= 500:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "auth service error")
    if resp.status_code != 200:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"auth rejected token ({resp.status_code})")

    profile = resp.json()
    ident = Identity(
        id=str(profile.get("id") or profile.get("user_id") or "").strip(),
        email=profile.get("email") if isinstance(profile.get("email"), str) else None,
        organization_id=str(profile["organization_id"]) if profile.get("organization_id") else None,
        is_active=bool(profile.get("is_active", True)),
    )
    if not ident.id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "auth profile missing id")
    if not ident.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "user is not active")

    _cache[token] = (now + settings.auth_verify_cache_ttl_s, ident)
    return ident


async def require_auth(request: Request, authorization: str | None = Header(default=None)) -> None:
    mode = settings.auth_mode
    set_principal(None)  # reset per request (open/token modes have no principal)
    if mode == "none":
        return

    token = _bearer(authorization)
    if mode == "token":
        if not token or token != settings.api_token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return

    # cyberdyne
    if not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    ident = await verify_cyberdyne(token)
    request.state.identity = ident
    set_principal(ident.id)


def current_identity(request: Request) -> Identity | None:
    return getattr(request.state, "identity", None)
