"""Workspace I/O — upload data in, download results out (TRD F5).

  * POST   /v1/files          multipart upload into the session workspace
  * GET    /v1/files          list the workspace tree
  * GET    /v1/files/{path}   stream a file back (content-type by ext)
"""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

import limits
import services
from config import settings
from models import FileInfo, FileListResponse, UploadResponse
from workspaces import list_files, resolve_in_workspace

router = APIRouter(prefix="/v1/files", tags=["files"])

_CHUNK = 1024 * 1024


@router.post("", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> UploadResponse:
    # Session-aware: while a stateful REPL session is live, this resolves to the
    # worker's adopted workspace, so uploads and REPL artifacts share one dir.
    ws = services.resolve_workspace(user_id, session_id)
    name = (file.filename or "").strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing filename")
    try:
        dest = resolve_in_workspace(ws, name)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid path")

    dest.parent.mkdir(parents=True, exist_ok=True)
    cap = settings.max_upload_mb * 1024 * 1024
    quota = settings.user_quota_mb * 1024 * 1024
    base = limits.dir_size(ws) if quota > 0 else 0
    written = 0
    with dest.open("wb") as out:
        while chunk := await file.read(_CHUNK):
            written += len(chunk)
            if written > cap:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,  # Content Too Large
                    f"upload exceeds {settings.max_upload_mb} MB",
                )
            if quota > 0 and base + written > quota:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,  # Content Too Large
                    f"workspace quota of {settings.user_quota_mb} MB exceeded",
                )
            out.write(chunk)

    st = dest.stat()
    rel = str(dest.relative_to(ws.resolve()))
    return UploadResponse(ok=True, file=FileInfo(path=rel, size=st.st_size, modified=st.st_mtime))


@router.get("", response_model=FileListResponse)
async def listing(
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> FileListResponse:
    ws = services.resolve_workspace(user_id, session_id)
    return FileListResponse(
        files=[FileInfo(path=e.path, size=e.size, modified=e.modified) for e in list_files(ws)]
    )


@router.get("/{path:path}")
async def download(
    path: str,
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> FileResponse:
    ws = services.resolve_workspace(user_id, session_id)
    try:
        target = resolve_in_workspace(ws, path)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid path")
    if not target.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file not found")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=target.name)
