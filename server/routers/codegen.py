"""POST /v1/codegen/{target} — transpile to one real emit flag each."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

import services
from models import CodegenRequest, CodegenResponse

router = APIRouter(prefix="/v1", tags=["codegen"])


@router.post("/codegen/{target}", response_model=CodegenResponse)
async def codegen(target: str, req: CodegenRequest) -> CodegenResponse:
    try:
        result = await services.run_codegen(target, req.source, req.user_id, req.session_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return CodegenResponse(**result)
