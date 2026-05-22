"""POST /v1/plot — run a plotting snippet and stream back the figure (TRD F5).

Format is chosen by (in priority order) the ``?format=`` query, the request
body ``format``, the ``Accept`` header, then the server default.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import FileResponse

import services
from config import settings
from models import PlotRequest

router = APIRouter(prefix="/v1", tags=["plot"])

_MIME = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
_ACCEPT = {"image/png": "png", "image/svg+xml": "svg", "application/pdf": "pdf"}


def _accept_format(accept: str | None) -> str | None:
    if not accept:
        return None
    for media, fmt in _ACCEPT.items():
        if media in accept:
            return fmt
    return None


@router.post("/plot")
async def plot(
    req: PlotRequest,
    format: str | None = Query(default=None),
    accept: str | None = Header(default=None),
) -> FileResponse:
    fmt = (format or req.format or _accept_format(accept) or settings.plot_default_format).lower()
    if fmt not in _MIME:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unsupported format {fmt!r}")

    res = await services.run_plot(req.source, fmt, req.user_id, req.session_id)
    if not res["ok"]:
        raise HTTPException(
            422,  # Unprocessable Content
            detail={"error": "no figure was produced", "stderr": res["stderr"], "timed_out": res["timed_out"]},
        )
    path = Path(res["file"])
    return FileResponse(path, media_type=_MIME[fmt], filename=path.name)
