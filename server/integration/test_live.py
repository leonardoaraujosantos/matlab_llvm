"""End-to-end tests against a live uvicorn server (real HTTP/WS/MCP).

Works with the real ``build/matlabc`` or the fake stub; assertions are kept
to behaviour that holds for both. Real-compiler-only paths are guarded by the
``is_real`` fixture.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest


def test_healthz(http):
    body = http.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["matlabc_present"] is True


def test_check_ok(http):
    r = http.post("/v1/check", json={"source": "x = 1 + 1;\ndisp(x)\n"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_repl_stateful_persists_across_calls(http):
    s = "live_persist"
    a = http.post("/v1/repl", json={"source": "x = 41;", "session_id": s}).json()
    assert a["stateful"] is True
    b = http.post("/v1/repl", json={"source": "disp(x)", "session_id": s}).json()
    assert b["ok"] is True
    assert "41" in b["stdout"]


def test_codegen_python(http):
    r = http.post("/v1/codegen/python", json={"source": "y = 2;\ndisp(y)\n"}).json()
    assert r["ok"] is True
    assert r["language"] == "python"
    assert r["code"].strip() != ""


def test_files_upload_download_roundtrip(http):
    content = b"1,2,3\n4,5,6\n"
    up = http.post(
        "/v1/files",
        params={"session_id": "live_files"},
        files={"file": ("data.csv", content, "text/csv")},
    )
    assert up.status_code == 200, up.text
    dl = http.get("/v1/files/data.csv", params={"session_id": "live_files"})
    assert dl.status_code == 200
    assert dl.content == content


def test_plot_png(http, is_real):
    r = http.post("/v1/plot", json={"source": "plot([1 2 3])", "session_id": "live_plot"})
    if is_real and r.status_code == 422:
        pytest.skip("real matlabc built without -DMATLAB_LLVM_WITH_PLOT")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")


def test_chat_offline_grounded(http):
    r = http.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What does fmincon do?"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "x_citations" in body


def test_mcp_over_http(base_url):
    from fastmcp import Client

    async def go():
        async with Client(base_url + "/mcp/") as c:
            return sorted(t.name for t in await c.list_tools())

    names = asyncio.run(go())
    assert {"matlab_check", "matlab_repl", "matlab_codegen"} <= set(names)


def test_dap_over_websocket(http, base_url):
    import websockets

    http.post(
        "/v1/files",
        params={"session_id": "live_dap"},
        files={"file": ("program.m", b"x = 1;\ndisp(x)\n", "text/plain")},
    )
    ws_url = base_url.replace("http", "ws") + "/v1/dap/ws/live_dap?program=program.m"

    async def go():
        async with websockets.connect(ws_url) as ws:
            req = {"seq": 1, "type": "request", "command": "initialize", "arguments": {"adapterID": "matlab"}}
            body = json.dumps(req).encode("utf-8")
            await ws.send(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            buf = b""
            for _ in range(10):
                data = await asyncio.wait_for(ws.recv(), timeout=15)
                buf += data if isinstance(data, bytes) else data.encode("utf-8")
                if b'"command"' in buf and b"initialize" in buf:
                    return buf
            return buf

    out = asyncio.run(go())
    assert b"initialize" in out
