"""POST /v1/repl — JIT-execute statements, capture text + new figures.

Phase 1 is stateless: every call is a fresh ``matlabc -repl`` child whose
cwd is the session workspace. Stateful long-lived sessions are a Phase-7
addition (see plan §11).
"""

from __future__ import annotations

from fastapi import APIRouter

import services
from models import ReplRequest, ReplResponse

router = APIRouter(prefix="/v1", tags=["repl"])


@router.post("/repl", response_model=ReplResponse)
async def repl(req: ReplRequest) -> ReplResponse:
    return ReplResponse(
        **await services.run_repl(req.source, req.user_id, req.session_id, req.stateful)
    )
