"""Core operations shared by the REST routers and the MCP tools.

Each function resolves the session workspace, drives the matlabc wrapper, and
returns plain JSON-friendly dicts (diagnostics as dicts, not models) so both
Pydantic responses and MCP serialization can consume them directly.
"""

from __future__ import annotations

import matlabc
from diagnostics import parse_diagnostics
from workspaces import (
    list_files,
    new_artifacts,
    resolve_in_workspace,
    snapshot,
    workspace_for,
)


def _diag_dicts(text: str) -> list[dict]:
    return [d.model_dump() for d in parse_diagnostics(text)]


async def run_check(source: str, user_id: str | None = None, session_id: str | None = None) -> dict:
    ws = workspace_for(user_id, session_id)
    res = await matlabc.check(source, ws)
    return {
        "ok": res.ok,
        "diagnostics": _diag_dicts(f"{res.stderr}\n{res.stdout}"),
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


async def run_repl(source: str, user_id: str | None = None, session_id: str | None = None) -> dict:
    ws = workspace_for(user_id, session_id)
    before = snapshot(ws)
    res = await matlabc.repl(source, ws)
    return {
        "ok": res.ok,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "timed_out": res.timed_out,
        "truncated": res.stdout_truncated or res.stderr_truncated,
        "artifacts": new_artifacts(ws, before),
    }


async def run_codegen(
    target: str, source: str, user_id: str | None = None, session_id: str | None = None
) -> dict:
    if target not in matlabc.EMIT_FLAGS:
        raise ValueError(
            f"unknown target {target!r}; expected one of {sorted(matlabc.EMIT_FLAGS)}"
        )
    ws = workspace_for(user_id, session_id)
    res = await matlabc.emit(target, source, ws)
    return {
        "ok": res.ok,
        "language": target,
        "code": res.stdout if res.ok else "",
        "diagnostics": _diag_dicts(res.stderr),
        "stderr": res.stderr,
    }


def list_workspace(user_id: str | None = None, session_id: str | None = None) -> list[dict]:
    ws = workspace_for(user_id, session_id)
    return [{"path": e.path, "size": e.size, "modified": e.modified} for e in list_files(ws)]


def read_workspace_file(
    path: str,
    user_id: str | None = None,
    session_id: str | None = None,
    max_bytes: int = 1_000_000,
) -> bytes:
    ws = workspace_for(user_id, session_id)
    target = resolve_in_workspace(ws, path)  # raises ValueError on traversal
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_bytes()[:max_bytes]
