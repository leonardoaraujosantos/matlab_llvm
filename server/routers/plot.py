"""POST /v1/plot — run a plotting snippet and return the produced figure (TRD F5).

By default the response is JSON (:class:`PlotResponse`), consistent with the
OpenAPI declaration and shaped like ``/v1/repl``: the figure path lands in
``artifacts`` and the bytes are fetched via ``GET /v1/files/{path}``. Generated
OpenAPI clients work without special-casing a binary body.

For convenience the raw figure bytes can still be streamed directly by passing
``?raw=true`` or an image/pdf ``Accept`` header.

Format is chosen by (in priority order) the ``?format=`` query, the request
body ``format``, the ``Accept`` header, then the server default.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import FileResponse

import services
from config import settings
from models import PlotRequest, PlotResponse

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


def _wants_raw(raw: bool, accept: str | None) -> bool:
    """Stream raw bytes when explicitly requested via ?raw or an image/pdf Accept.

    A generic OpenAPI client sends ``Accept: application/json`` (or ``*/*``) and
    gets JSON; only a caller that explicitly asks for an image/pdf media type
    (or sets ``?raw=true``) gets the binary body.
    """
    if raw:
        return True
    return _accept_format(accept) is not None


@router.post("/plot", response_model=PlotResponse)
async def plot(
    req: PlotRequest,
    format: str | None = Query(default=None),
    raw: bool = Query(default=False, description="Stream raw figure bytes instead of JSON."),
    accept: str | None = Header(default=None),
):
    fmt = (format or req.format or _accept_format(accept) or settings.plot_default_format).lower()
    if fmt not in _MIME:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unsupported format {fmt!r}")

    res = await services.run_plot(req.source, fmt, req.user_id, req.session_id)
    if not res["ok"]:
        raise HTTPException(
            422,  # Unprocessable Content
            detail={"error": "no figure was produced", "stderr": res["stderr"], "timed_out": res["timed_out"]},
        )

    if _wants_raw(raw, accept):
        path = Path(res["file"])
        return FileResponse(path, media_type=_MIME[fmt], filename=path.name)

    return PlotResponse(
        ok=res["ok"],
        format=fmt,
        stdout=res["stdout"],
        stderr=res["stderr"],
        timed_out=res["timed_out"],
        truncated=res["truncated"],
        artifacts=res["artifacts"],
    )
