"""WS /v1/dap/ws/{session_id} — DAP-over-WebSocket bridge (plan Phase 4).

Spawns ``matlabc -dap <workspace/program>`` confined to the sandbox and runs
two pumps, preserving the DAP ``Content-Length`` framing as an opaque byte
stream (the client owns DAP semantics). One child per connection, killed on
disconnect/EOF. Child stderr is surfaced as text "diagnostic" messages so it
never pollutes the binary DAP stream.

The matlabc DAP server JIT-executes the program in-process, so the same
sandbox limits apply and a crash dies with the child, never the bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

import sandbox
from auth import verify_cyberdyne
from config import settings
from principal import effective_user, set_principal
from workspaces import resolve_in_workspace, workspace_for

log = logging.getLogger("matlab_backend.dap")
router = APIRouter()

_CHUNK = 65536


@router.websocket("/v1/dap/ws/{session_id}")
async def dap_ws(
    websocket: WebSocket,
    session_id: str,
    program: str = Query("program.m", description="Program file in the workspace to debug."),
    user_id: str | None = Query(None),
    token: str | None = Query(None, description="Bearer token (WS can't set Authorization)."),
) -> None:
    await websocket.accept()

    # Auth via query param (browsers can't set Authorization on a WS handshake).
    mode = settings.auth_mode
    if mode == "token":
        if not token or token != settings.api_token:
            await websocket.close(code=1008, reason="unauthorized")
            return
    elif mode == "cyberdyne":
        if not token:
            await websocket.close(code=1008, reason="unauthorized")
            return
        try:
            ident = await verify_cyberdyne(token)
        except HTTPException as exc:
            code = 1008 if exc.status_code in (401, 403) else 1011
            await websocket.close(code=code, reason="unauthorized")
            return
        set_principal(ident.id)

    ws_dir = workspace_for(effective_user(user_id), session_id)
    try:
        prog = resolve_in_workspace(ws_dir, program)
    except ValueError:
        await websocket.close(code=1008, reason="invalid program path")
        return
    if not prog.is_file():
        await websocket.close(code=1008, reason=f"program not found: {program}")
        return
    if not settings.matlabc_path.exists():
        await websocket.close(code=1011, reason="matlabc not available")
        return

    proc = await sandbox.spawn([str(settings.matlabc_path), "-dap", str(prog)], cwd=ws_dir)
    log.info("dap session %s: spawned pid=%s for %s", session_id, proc.pid, prog)

    async def ws_to_child() -> None:
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None and msg.get("text") is not None:
                    data = msg["text"].encode("utf-8")
                if data:
                    proc.stdin.write(data)
                    await proc.stdin.drain()
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
            pass
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                try:
                    proc.stdin.close()
                except Exception:
                    pass

    async def child_to_ws() -> None:
        try:
            while True:
                chunk = await proc.stdout.read(_CHUNK)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
            pass

    async def stderr_to_ws() -> None:
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                await websocket.send_text(
                    json.dumps(
                        {"type": "diagnostic", "stream": "stderr",
                         "data": line.decode("utf-8", "replace")}
                    )
                )
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
            pass

    tasks = [asyncio.create_task(p()) for p in (ws_to_child, child_to_ws, stderr_to_ws)]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Unwind the pumps gracefully rather than hard-cancelling: killing the
        # child makes the stdout/stderr reads hit EOF, and closing the socket
        # makes a pump blocked on receive()/send() return a disconnect. A bare
        # task.cancel() of a receive() upsets Starlette's test portal.
        sandbox.terminate(proc)
        try:
            await websocket.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3.0)
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        log.info("dap session %s: closed (rc=%s)", session_id, proc.returncode)
