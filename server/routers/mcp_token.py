"""POST /v1/mcp/token — mint an MCP bearer token for the caller.

Authenticated like every /v1 route (so in cyberdyne mode it's bound to the
verified identity). The MCP client then presents this token on /mcp; it maps
back to the same user, so MCP tools run against that user's workspace.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

import mcp_auth
from models import McpTokenResponse

router = APIRouter(prefix="/v1", tags=["mcp"])


@router.post("/mcp/token", response_model=McpTokenResponse)
async def create_mcp_token(request: Request) -> McpTokenResponse:
    ident = getattr(request.state, "identity", None)
    subject = ident.id if ident is not None else "anon"
    email = ident.email if ident is not None else None
    token, exp = mcp_auth.mint(subject, email)
    return McpTokenResponse(token=token, subject=subject, expires_at=exp)
