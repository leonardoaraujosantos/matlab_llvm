"""GET /v1/auth/whoami — echo the authenticated identity (auth smoke).

Goes through ``require_auth`` like every /v1 route: in cyberdyne/token mode a
valid bearer is required; in open mode it reports ``authenticated: false``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from config import settings
from models import WhoAmIResponse

router = APIRouter(prefix="/v1", tags=["auth"])


@router.get("/auth/whoami", response_model=WhoAmIResponse)
async def whoami(request: Request) -> WhoAmIResponse:
    ident = getattr(request.state, "identity", None)
    if ident is None:
        return WhoAmIResponse(authenticated=False, mode=settings.auth_mode)
    return WhoAmIResponse(
        authenticated=True,
        mode=settings.auth_mode,
        id=ident.id,
        email=ident.email,
        organization_id=ident.organization_id,
    )
