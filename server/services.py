"""Core operations shared by the REST routers and the MCP tools.

Each function resolves the session workspace, drives the matlabc wrapper, and
returns plain JSON-friendly dicts (diagnostics as dicts, not models) so both
Pydantic responses and MCP serialization can consume them directly.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import matlabc
from config import settings
from diagnostics import parse_diagnostics
from principal import effective_user
from sessions import MANAGER
from workspaces import (
    list_files,
    new_artifacts,
    resolve_in_workspace,
    snapshot,
    workspace_for,
)

_PLOT_EXT = ("png", "svg", "pdf")


def resolve_workspace(user_id: str | None = None, session_id: str | None = None) -> Path:
    """A live stateful session's bound workspace, else the deterministic one.
    The authenticated principal (if any) overrides the client-supplied id."""
    user_id = effective_user(user_id)
    return MANAGER.workspace_of(user_id, session_id) or workspace_for(user_id, session_id)


def _diag_dicts(text: str) -> list[dict]:
    return [d.model_dump() for d in parse_diagnostics(text)]


async def run_check(source: str, user_id: str | None = None, session_id: str | None = None) -> dict:
    ws = resolve_workspace(user_id, session_id)
    res = await matlabc.check(source, ws)
    return {
        "ok": res.ok,
        "diagnostics": _diag_dicts(f"{res.stderr}\n{res.stdout}"),
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


async def run_repl(
    source: str,
    user_id: str | None = None,
    session_id: str | None = None,
    stateful: bool | None = None,
) -> dict:
    use_stateful = settings.repl_stateful if stateful is None else stateful
    user_id = effective_user(user_id)  # isolate sessions per authenticated identity

    if use_stateful:
        res = await MANAGER.run_turn(user_id, session_id, source, settings.wall_timeout_s)
        out = res["stdout"]
        truncated = len(out.encode("utf-8")) > settings.output_cap_bytes
        if truncated:
            out = out[: settings.output_cap_bytes]
        return {
            "ok": (not res["timed_out"]) and res["alive"],
            "stdout": out,
            "stderr": "",  # merged into stdout for stateful turns
            "timed_out": res["timed_out"],
            "truncated": truncated,
            "artifacts": res["artifacts"],
            "stateful": True,
        }

    ws = resolve_workspace(user_id, session_id)
    before = snapshot(ws)
    res = await matlabc.repl(source, ws)
    return {
        "ok": res.ok,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "timed_out": res.timed_out,
        "truncated": res.stdout_truncated or res.stderr_truncated,
        "artifacts": new_artifacts(ws, before),
        "stateful": False,
    }


async def run_codegen(
    target: str, source: str, user_id: str | None = None, session_id: str | None = None
) -> dict:
    if target not in matlabc.EMIT_FLAGS:
        raise ValueError(
            f"unknown target {target!r}; expected one of {sorted(matlabc.EMIT_FLAGS)}"
        )
    ws = resolve_workspace(user_id, session_id)
    res = await matlabc.emit(target, source, ws)
    return {
        "ok": res.ok,
        "language": target,
        "code": res.stdout if res.ok else "",
        "diagnostics": _diag_dicts(res.stderr),
        "stderr": res.stderr,
    }


def list_workspace(user_id: str | None = None, session_id: str | None = None) -> list[dict]:
    ws = resolve_workspace(user_id, session_id)
    return [{"path": e.path, "size": e.size, "modified": e.modified} for e in list_files(ws)]


def read_workspace_file(
    path: str,
    user_id: str | None = None,
    session_id: str | None = None,
    max_bytes: int = 1_000_000,
) -> bytes:
    ws = resolve_workspace(user_id, session_id)
    target = resolve_in_workspace(ws, path)  # raises ValueError on traversal
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_bytes()[:max_bytes]


async def run_plot(
    source: str,
    fmt: str = "png",
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Run a plotting snippet and save the current figure to the workspace.

    A self-contained one-shot run: the server appends ``saveas(gcf, OUT)``
    with an *absolute* OUT path, so the figure lands in the workspace
    regardless of the child's cwd. Returns the produced file or an error.
    """
    if fmt not in _PLOT_EXT:
        raise ValueError(f"unknown format {fmt!r}; expected one of {list(_PLOT_EXT)}")
    ws = resolve_workspace(user_id, session_id)
    out = ws / f"plot_{uuid.uuid4().hex[:12]}.{fmt}"
    snippet = source.rstrip("\n") + f"\nsaveas(gcf, '{out}');\n"
    res = await matlabc.repl(snippet, ws)
    produced = out.is_file() and out.stat().st_size > 0
    return {
        "ok": produced and not res.timed_out,
        "file": str(out),
        "rel": out.name,
        "stderr": res.stderr,
        "timed_out": res.timed_out,
    }
