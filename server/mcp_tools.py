"""FastMCP tools exposing the compiler to MCP/AI clients (plan Phase 5).

Thin wrappers over the shared :mod:`services` layer, so MCP and REST share
the same sandbox + workspace isolation. Mounted into the FastAPI app in
``main.py`` (streamable-HTTP at ``/mcp``).

NOTE: this module is deliberately NOT named ``mcp`` — a local ``mcp`` package
would shadow the installed ``mcp`` SDK that fastmcp imports.
"""

from __future__ import annotations

from fastmcp import FastMCP

import services
from matlabc import EMIT_FLAGS

mcp_server = FastMCP("matlab_llvm")


@mcp_server.tool
async def matlab_check(source: str, session_id: str | None = None) -> dict:
    """Validate MATLAB source without executing it. Returns {ok, diagnostics}."""
    return await services.run_check(source, session_id=session_id)


@mcp_server.tool
async def matlab_repl(source: str, session_id: str | None = None) -> dict:
    """JIT-execute MATLAB statements; returns stdout/stderr and figure artifacts."""
    return await services.run_repl(source, session_id=session_id)


@mcp_server.tool
async def matlab_codegen(target: str, source: str, session_id: str | None = None) -> dict:
    """Transpile MATLAB to one target language.

    target: one of python, typescript, c, cpp, systemverilog.
    """
    return await services.run_codegen(target, source, session_id=session_id)


@mcp_server.tool
def list_files(session_id: str | None = None) -> list[dict]:
    """List files in the session workspace (uploaded data, results, figures)."""
    return services.list_workspace(session_id=session_id)


@mcp_server.tool
def read_file(path: str, session_id: str | None = None) -> str:
    """Read a UTF-8 text file from the session workspace."""
    return services.read_workspace_file(path, session_id=session_id).decode("utf-8", "replace")


# Advertised codegen targets, handy for clients building the tool call.
CODEGEN_TARGETS = sorted(EMIT_FLAGS)
