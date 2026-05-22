"""POST /v1/check — validate-only. The single sub-200ms route."""

from __future__ import annotations

from fastapi import APIRouter

import services
from models import CheckRequest, CheckResponse

router = APIRouter(prefix="/v1", tags=["check"])


@router.post("/check", response_model=CheckResponse)
async def check(req: CheckRequest) -> CheckResponse:
    return CheckResponse(**await services.run_check(req.source, req.user_id, req.session_id))
