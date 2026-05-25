"""End-to-end tests against a live uvicorn server (real HTTP/WS/MCP).

Runs against either a locally-booted ``uvicorn`` subprocess (default) or a
remote deployment when ``BACKEND_URL`` is set — see ``conftest.py``. Both
modes share the same fixtures, so the same assertions apply.

Real-compiler-only paths are guarded by the ``is_real`` fixture; the plot
test additionally checks ``plot_supported`` (matlabc built with Cairo).
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest


def _session_id(label: str) -> str:
    """Unique session id per test run so remote runs don't reuse stale state."""
    return f"itest-{label}-{uuid.uuid4().hex[:8]}"


def test_healthz(http):
    r = http.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["matlabc_present"] is True


def test_healthz_sandbox_field(http):
    """The deployment exposes its tier-2 sandbox state."""
    body = http.get("/healthz").json()
    assert "sandbox" in body, body
    sb = body["sandbox"]
    assert sb["backend"] in ("none", "bwrap", "firejail", "nsjail")
    assert isinstance(sb["active"], bool)


def test_sandbox_matches_expected(http):
    """If EXPECT_SANDBOX=<backend> is set, the deployment MUST report it active.

    Use this in CI / post-deploy to fail loudly when sandbox flips back to
    'none' (e.g. someone removed the env var on Coolify). When EXPECT_SANDBOX
    is unset the test is a no-op.
    """
    import os

    expected = os.environ.get("EXPECT_SANDBOX", "").strip()
    if not expected:
        pytest.skip("EXPECT_SANDBOX not set — skipping post-deploy guardrail")
    sb = http.get("/healthz").json().get("sandbox", {})
    assert sb.get("backend") == expected, sb
    assert sb.get("active") is True, (
        f"sandbox configured as {expected!r} but not active "
        f"(tool missing or userns blocked): {sb}"
    )


def test_whoami(http, auth_token):
    r = http.get("/v1/auth/whoami")
    assert r.status_code == 200, r.text
    body = r.json()
    if auth_token:
        assert body.get("authenticated") is True
        # The identity is exposed as `id` (and `email` if known).
        assert body.get("id")
    else:
        # Open / no-auth mode is allowed locally.
        assert "authenticated" in body


def test_check_ok(http):
    r = http.post("/v1/check", json={"source": "x = 1 + 1;\ndisp(x)\n"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


def test_check_syntax_error(http):
    r = http.post("/v1/check", json={"source": "x = ;\n"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Either ok=false or diagnostics non-empty — the API surfaces problems.
    assert body["ok"] is False or body.get("diagnostics")


def test_repl_disp(http):
    r = http.post("/v1/repl", json={"source": "disp(6*7)", "session_id": _session_id("disp")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "42" in body["stdout"]


def test_repl_stateful_persists_across_calls(http):
    sid = _session_id("persist")
    a = http.post("/v1/repl", json={"source": "x = 41;", "session_id": sid}).json()
    assert a["stateful"] is True
    b = http.post("/v1/repl", json={"source": "disp(x)", "session_id": sid}).json()
    assert b["ok"] is True
    assert "41" in b["stdout"]


_SOFT_SOURCE = "y = 2;\ndisp(y)\n"
# SystemVerilog only synthesizes typed functions — mirrors test/EmitSV/add_scalar.m.
_SV_SOURCE = (
    "T = numerictype(1, 16, 0);\n"
    "y = add_scalar(fi(3, T), fi(4, T));\n"
    "disp(y);\n"
    "function y = add_scalar(a, b)\n"
    "    y = a + b;\n"
    "end\n"
)


@pytest.mark.parametrize(
    "target,language,source",
    [
        ("python", "python", _SOFT_SOURCE),
        ("typescript", "typescript", _SOFT_SOURCE),
        ("c", "c", _SOFT_SOURCE),
        ("cpp", "cpp", _SOFT_SOURCE),
        ("systemverilog", "systemverilog", _SV_SOURCE),
    ],
)
def test_codegen_targets(http, target, language, source):
    r = http.post(f"/v1/codegen/{target}", json={"source": source})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    assert body["language"] == language
    assert body["code"].strip() != ""


def test_files_upload_list_download_roundtrip(http):
    sid = _session_id("files")
    content = b"1,2,3\n4,5,6\n"
    up = http.post(
        "/v1/files",
        params={"session_id": sid},
        files={"file": ("data.csv", content, "text/csv")},
    )
    assert up.status_code == 200, up.text

    lst = http.get("/v1/files", params={"session_id": sid})
    assert lst.status_code == 200, lst.text
    paths = [entry["path"] for entry in lst.json().get("files", [])]
    assert "data.csv" in paths

    dl = http.get("/v1/files/data.csv", params={"session_id": sid})
    assert dl.status_code == 200, dl.text
    assert dl.content == content


def test_files_rejects_traversal(http):
    sid = _session_id("traverse")
    # Try to read outside the workspace — must be rejected.
    r = http.get("/v1/files/..%2Fetc%2Fpasswd", params={"session_id": sid})
    assert r.status_code in (400, 403, 404)


def test_plot_png(http, is_real, plot_supported):
    if not is_real:
        pytest.skip("plot needs the real matlabc, not the fake stub")
    if not plot_supported:
        pytest.skip("matlabc built without -DMATLAB_LLVM_WITH_PLOT")
    sid = _session_id("plot")
    r = http.post("/v1/plot", json={"source": "plot([1 2 3])", "session_id": sid})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    assert len(r.content) > 100  # PNG header alone is 8 bytes; real plots are bigger.


def test_chat_completion_structure(http):
    """Works in three modes:

    * retrieval-only (no ``OPENAI_API_KEY``): 200, full chat.completion body
    * OpenAI proxy success: 200, full chat.completion body
    * OpenAI proxy failure (e.g. invalid key): 4xx, still grounded via
      ``x_citations`` — proves the RAG layer is alive

    All three are acceptable signals that the endpoint is wired correctly.
    """
    r = http.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What does fmincon do?"}]},
    )
    assert r.status_code in (200, 401, 403, 429, 502, 503), r.text
    body = r.json()
    # Citations are always attached by the backend regardless of upstream.
    assert "x_citations" in body
    assert isinstance(body["x_citations"], list) and len(body["x_citations"]) > 0
    if r.status_code == 200:
        assert body.get("object") == "chat.completion"
        assert body["choices"][0]["message"]["content"].strip() != ""


def test_mcp_token_mint(http):
    r = http.post("/v1/mcp/token")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("token")
    assert isinstance(body["token"], str) and len(body["token"]) > 20


def test_mcp_over_http_with_minted_token(http, base_url):
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    token = http.post("/v1/mcp/token").json()["token"]

    async def go():
        async with Client(base_url + "/mcp/", auth=BearerAuth(token)) as c:
            return sorted(t.name for t in await c.list_tools())

    names = asyncio.run(go())
    assert {"matlab_check", "matlab_repl", "matlab_codegen"} <= set(names)


def test_mcp_over_http_rejects_without_token(base_url, mode):
    if mode == "local":
        # Local boot turns MCP auth on, so this always rejects. Verified.
        pass
    # On remote we also turned MCP auth on (MATLAB_BACKEND_MCP_REQUIRE_AUTH=1).
    from fastmcp import Client

    async def go():
        async with Client(base_url + "/mcp/") as c:
            await c.list_tools()

    with pytest.raises(Exception):
        asyncio.run(go())


def test_dap_over_websocket(http, base_url, ws_auth_qs):
    import websockets

    sid = _session_id("dap")
    http.post(
        "/v1/files",
        params={"session_id": sid},
        files={"file": ("program.m", b"x = 1;\ndisp(x)\n", "text/plain")},
    )
    ws_base = base_url.replace("http", "ws", 1)
    ws_url = f"{ws_base}/v1/dap/ws/{sid}?{ws_auth_qs}program=program.m"

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
