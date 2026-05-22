"""MCP tool tests via the FastMCP in-memory client (no network/transport).

Tools wrap the same services as REST, so the fake matlabc (conftest.py)
drives them too.
"""

import asyncio
import io


def _run(coro):
    return asyncio.run(coro)


def _client():
    from fastmcp import Client

    from mcp_tools import mcp_server

    return Client(mcp_server)


def test_mcp_lists_expected_tools():
    async def go():
        async with _client() as c:
            return sorted(t.name for t in await c.list_tools())

    names = set(_run(go()))
    assert {"matlab_check", "matlab_repl", "matlab_codegen", "list_files", "read_file"} <= names


def test_mcp_repl_and_codegen():
    async def go():
        async with _client() as c:
            repl = await c.call_tool("matlab_repl", {"source": "disp(7)", "session_id": "mcpsess"})
            cg = await c.call_tool(
                "matlab_codegen", {"target": "python", "source": "y = 2;", "session_id": "mcpsess"}
            )
            return repl.data, cg.data

    repl, cg = _run(go())
    assert repl["ok"] is True
    assert "7" in repl["stdout"]
    assert cg["ok"] is True
    assert cg["language"] == "python"
    assert "print" in cg["code"]


def test_mcp_codegen_bad_target_errors():
    async def go():
        async with _client() as c:
            try:
                await c.call_tool("matlab_codegen", {"target": "rust", "source": "y = 2;"})
                return "no-error"
            except Exception as exc:  # FastMCP raises a ToolError on failure
                return type(exc).__name__

    assert _run(go()) != "no-error"


def test_mcp_list_and_read_files(client):
    client.post(
        "/v1/files",
        params={"session_id": "mcpfiles"},
        files={"file": ("hello.txt", io.BytesIO(b"hi there"), "text/plain")},
    )

    async def go():
        async with _client() as c:
            listing = (await c.call_tool("list_files", {"session_id": "mcpfiles"})).data
            content = (await c.call_tool("read_file", {"path": "hello.txt", "session_id": "mcpfiles"})).data
            return listing, content

    listing, content = _run(go())
    assert any(f["path"] == "hello.txt" for f in listing)
    assert content == "hi there"
