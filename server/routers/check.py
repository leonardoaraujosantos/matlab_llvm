"""POST /v1/check — validate-only. The single sub-200ms route."""

from __future__ import annotations

from fastapi import APIRouter

import matlabc
from diagnostics import parse_diagnostics
from models import CheckRequest, CheckResponse
from workspaces import workspace_for

router = APIRouter(prefix="/v1", tags=["check"])


@router.post("/check", response_model=CheckResponse)
async def check(req: CheckRequest) -> CheckResponse:
    ws = workspace_for(req.user_id, req.session_id)
    res = await matlabc.check(req.source, ws)
    return CheckResponse(
        ok=res.ok,
        diagnostics=parse_diagnostics(f"{res.stderr}\n{res.stdout}"),
        stdout=res.stdout,
        stderr=res.stderr,
    )
