"""Optional bearer-token auth.

When ``MATLAB_BACKEND_API_TOKEN`` is empty (the local-dev default) the
dependency is a no-op. Set it in production to require
``Authorization: Bearer <token>`` on every /v1 and /mcp route.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from config import settings


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
