"""Per-user / per-session workspace resolution.

A workspace is a directory under ``WORKSPACE_ROOT`` that becomes the cwd
of every child process for that session: uploaded data, generated code,
saved ``.mat``/CSV files and figures all live here. Identifiers are
sanitised so they can never escape the root.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from config import settings

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe(part: str | None, default: str) -> str:
    part = (part or "").strip()
    part = _UNSAFE.sub("_", part)
    part = part.strip(".")  # never "." / ".." / leading-dot tricks
    return part[:64] or default


def workspace_for(user_id: str | None = None, session_id: str | None = None) -> Path:
    """Resolve (and create) the workspace for a user/session pair."""
    ws = settings.workspace_root_path / _safe(user_id, "anon") / _safe(session_id, "default")
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def resolve_in_workspace(ws: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``ws``, rejecting path traversal."""
    ws = ws.resolve()
    target = (ws / rel).resolve()
    if target != ws and ws not in target.parents:
        raise ValueError(f"path escapes workspace: {rel!r}")
    return target


@dataclass
class FileEntry:
    path: str       # relative to the workspace
    size: int       # bytes
    modified: float  # epoch seconds


def list_files(ws: Path) -> list[FileEntry]:
    """List every regular file in the workspace tree (relative paths)."""
    ws = ws.resolve()
    out: list[FileEntry] = []
    for root, _dirs, names in os.walk(ws):
        for name in names:
            p = Path(root) / name
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(FileEntry(path=str(p.relative_to(ws)), size=st.st_size, modified=st.st_mtime))
    out.sort(key=lambda e: e.path)
    return out


def snapshot(ws: Path) -> dict[str, float]:
    """Map relative path -> mtime, for diffing new artifacts after a run."""
    return {e.path: e.modified for e in list_files(ws)}


# Extensions a REPL/plot turn may newly produce that we surface as artifacts.
_ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".bmp"}


def new_artifacts(ws: Path, before: dict[str, float]) -> list[str]:
    """Image/figure files created or modified since ``before``."""
    out = []
    for e in list_files(ws):
        if Path(e.path).suffix.lower() not in _ARTIFACT_EXTS:
            continue
        if e.path not in before or e.modified > before[e.path] + 1e-6:
            out.append(e.path)
    return sorted(out)
