"""POST /v1/codegen/{target} — transpile to one real emit flag each."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

import matlabc
from diagnostics import parse_diagnostics
from models import CodegenRequest, CodegenResponse
from workspaces import workspace_for

router = APIRouter(prefix="/v1", tags=["codegen"])


@router.post("/codegen/{target}", response_model=CodegenResponse)
async def codegen(target: str, req: CodegenRequest) -> CodegenResponse:
    if target not in matlabc.EMIT_FLAGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown target {target!r}; expected one of {sorted(matlabc.EMIT_FLAGS)}",
        )
    ws = workspace_for(req.user_id, req.session_id)
    res = await matlabc.emit(target, req.source, ws)
    return CodegenResponse(
        ok=res.ok,
        language=target,
        code=res.stdout if res.ok else "",
        diagnostics=parse_diagnostics(res.stderr),
        stderr=res.stderr,
    )
