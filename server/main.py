"""FastAPI app factory for the matlab_llvm remote backend.

Run locally:  ``just backend-up``  (builds matlabc, then serves on :8000)
Run directly: ``cd server && uv run uvicorn main:app``
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

import rag
import sessions
from auth import require_auth
from config import settings
from limits import rate_limit
from mcp_tools import mcp_server
from routers import chat, check, codegen, dap_ws, files, repl

log = logging.getLogger("matlab_backend")

# MCP streamable-HTTP sub-app, mounted at /mcp (endpoint: /mcp/). Its session
# manager has its own lifespan that the parent app MUST run — see lifespan().
mcp_app = mcp_server.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.workspace_root_path.mkdir(parents=True, exist_ok=True)
    mc = settings.matlabc_path
    if mc.exists():
        log.info("matlabc: %s", mc)
    else:
        log.warning(
            "matlabc not found at %s — build it with `just build` or set "
            "MATLAB_BACKEND_MATLABC_BIN. Routes will fail until it exists.",
            mc,
        )
    if settings.rag_enabled:
        try:
            n = rag.build_default_index()
            log.info("RAG index: %d chunks from %s/docs", n, settings.source_context_root)
        except Exception as exc:  # never block startup on a corpus problem
            log.warning("RAG index build failed: %s", exc)
    # Background sweep for idle stateful REPL sessions.
    evict_task = asyncio.create_task(sessions.eviction_loop())
    try:
        # Nest the MCP session-manager lifespan inside ours.
        async with mcp_app.lifespan(app):
            yield
    finally:
        evict_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await evict_task
        await sessions.MANAGER.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="matlab_llvm remote backend",
        version="0.1.0",
        summary="MATLAB in your pocket — check, REPL, codegen and files over the matlabc CLI.",
        lifespan=lifespan,
    )

    protected = [Depends(require_auth), Depends(rate_limit)]
    for module in (check, repl, codegen, files, chat):
        app.include_router(module.router, dependencies=protected)
    # WS bridge: auth is enforced inside the handler via a `?token=` query
    # param (browsers can't set Authorization on a WebSocket handshake).
    app.include_router(dap_ws.router)
    # MCP tools over streamable-HTTP at /mcp/ (in-process client also works).
    app.mount("/mcp", mcp_app)

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict:
        return {
            "status": "ok",
            "matlabc": str(settings.matlabc_path),
            "matlabc_present": settings.matlabc_path.exists(),
        }

    return app


app = create_app()
