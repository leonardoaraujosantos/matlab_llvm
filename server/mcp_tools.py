"""FastMCP tools exposing the compiler to MCP/AI clients (plan Phase 5).

Thin wrappers over the shared :mod:`services` layer, so MCP and REST share
the same sandbox + workspace isolation. Mounted into the FastAPI app in
``main.py`` (streamable-HTTP at ``/mcp``).

NOTE: this module is deliberately NOT named ``mcp`` — a local ``mcp`` package
would shadow the installed ``mcp`` SDK that fastmcp imports.
"""

from __future__ import annotations

import mcp_auth
import services
from config import settings
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from matlabc import EMIT_FLAGS
from principal import set_principal


class McpTokenVerifier(TokenVerifier):
    """Verify backend-minted MCP bearer tokens (HMAC; see mcp_auth)."""

    async def verify_token(self, token: str) -> AccessToken | None:
        payload = mcp_auth.verify(token)
        if payload is None:
            return None
        sub = str(payload.get("sub") or "")
        if not sub:
            return None
        return AccessToken(token=token, client_id=sub, scopes=["mcp"], expires_at=int(payload["exp"]))


# Require a minted token on /mcp when configured; otherwise open (local dev,
# in-process tests). Bound at import — settings are fixed per process.
mcp_server = FastMCP("matlab_llvm", auth=McpTokenVerifier() if settings.mcp_require_auth else None)


def _bind_principal() -> None:
    """Scope this MCP request to the authenticated identity, so tools use that
    user's workspace. No-op when MCP auth is disabled."""
    try:
        token = get_access_token()
    except Exception:
        token = None
    if token is not None and getattr(token, "client_id", None):
        set_principal(token.client_id)


@mcp_server.tool
async def matlab_check(source: str, session_id: str | None = None) -> dict:
    """Validate MATLAB source without executing it. Returns {ok, diagnostics}."""
    _bind_principal()
    return await services.run_check(source, session_id=session_id)


@mcp_server.tool
async def matlab_repl(source: str, session_id: str | None = None) -> dict:
    """JIT-execute MATLAB statements; returns stdout/stderr and figure artifacts."""
    _bind_principal()
    return await services.run_repl(source, session_id=session_id)


@mcp_server.tool
async def matlab_codegen(target: str, source: str, session_id: str | None = None) -> dict:
    """Transpile MATLAB to one target language.

    target: one of python, typescript, c, cpp, systemverilog.
    """
    _bind_principal()
    return await services.run_codegen(target, source, session_id=session_id)


@mcp_server.tool
def list_files(session_id: str | None = None) -> list[dict]:
    """List files in the session workspace (uploaded data, results, figures)."""
    _bind_principal()
    return services.list_workspace(session_id=session_id)


@mcp_server.tool
def read_file(path: str, session_id: str | None = None) -> str:
    """Read a UTF-8 text file from the session workspace."""
    _bind_principal()
    return services.read_workspace_file(path, session_id=session_id).decode("utf-8", "replace")


# Advertised codegen targets, handy for clients building the tool call.
CODEGEN_TARGETS = sorted(EMIT_FLAGS)
