# matlab_llvm — remote backend

A FastAPI edge over the `matlabc` CLI: validate, JIT-run (REPL), transpile,
and move files in/out — "MATLAB in your pocket". Companion to
[`docs/remote_backend_plan.md`](../docs/remote_backend_plan.md) and
[`docs/remote_backend_trd.md`](../docs/remote_backend_trd.md).

## Status

Phases implemented on this branch:

- **Phase 1** — `/v1/check`, `/v1/repl`, hardened sandbox launcher.
- **Phase 2** — `/v1/codegen/{python,typescript,c,cpp,systemverilog}`.
- **Phase 3 (partial)** — workspaces + `/v1/files` upload/list/download and
  REPL figure-capture artifacts. (Dedicated `/v1/plot` deferred.)
- **Phase 4** — `WS /v1/dap/ws/{session_id}`: DAP-over-WebSocket bridge to
  `matlabc -dap`, opaque byte-stream passthrough, one child per connection.
- **Phase 5** — FastMCP tools (`matlab_check/repl/codegen`, `list_files`,
  `read_file`) mounted at `/mcp` (streamable-HTTP).
- **Phase 6** — `/v1/chat/completions` (OpenAI-compatible), grounded in a
  dependency-free BM25 index over `docs/**/*.md`; proxies to OpenAI when a
  key is set, else returns a retrieval-only answer.
- **Phase 0/8** — `Dockerfile` + `docker-compose.yaml` at the repo root.

Deferred: auth/quotas/warm-pool hardening (Phase 7), stateful sessions,
dedicated `/v1/plot`.

> The plan named the MCP mount `/mcp/sse`; SSE transport is deprecated in
> MCP, so this ships modern **streamable-HTTP** at `/mcp` instead.

## Secrets / `.env`

The OpenAI proxy reads `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`,
`OPENAI_MODEL`) from the environment or a **gitignored** `server/.env`.
Without a key, `/v1/chat/completions` runs in retrieval-only mode.

## Run it locally

From the repo root (needs [`just`](https://github.com/casey/just) and
[`uv`](https://docs.astral.sh/uv/)):

```sh
just backend-up          # builds matlabc, serves on :8000
just backend-up 9000     # custom port
just backend-dev         # same, with --reload for editing the server
just backend-test        # test suite (fake matlabc — no LLVM build needed)
```

Then open <http://localhost:8000/docs> (Swagger UI) or:

```sh
curl localhost:8000/healthz
curl -X POST localhost:8000/v1/check  -H 'content-type: application/json' \
     -d '{"source":"x = 1 + 1;"}'
curl -X POST localhost:8000/v1/repl   -H 'content-type: application/json' \
     -d '{"source":"disp(1+1)"}'
curl -X POST localhost:8000/v1/codegen/python -H 'content-type: application/json' \
     -d '{"source":"y = 2;"}'
```

Without `just`: `cd server && uv run uvicorn main:app` (set
`MATLAB_BACKEND_MATLABC_BIN` to your matlabc binary).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/healthz` | liveness + matlabc presence |
| POST | `/v1/check` | validate-only (default matlabc mode) |
| POST | `/v1/repl` | JIT-execute; returns stdout/stderr + figure artifacts |
| POST | `/v1/codegen/{target}` | transpile to python/typescript/c/cpp/systemverilog |
| POST | `/v1/files` | multipart upload into the session workspace |
| GET  | `/v1/files` | list the workspace tree |
| GET  | `/v1/files/{path}` | download a file |
| WS   | `/v1/dap/ws/{session_id}` | DAP-over-WebSocket bridge (`?program=`, `?token=`) |
| MCP  | `/mcp` | FastMCP tools over streamable-HTTP for AI clients |
| POST | `/v1/chat/completions` | OpenAI-compatible chat grounded in the docs (RAG) |

Request bodies accept optional `user_id` / `session_id` (files endpoints take
them as query params) which select the workspace directory.

## Configuration

Env vars, prefix `MATLAB_BACKEND_` (or a `.env` file). See `config.py`.

| Var | Default | Meaning |
|---|---|---|
| `MATLABC_BIN` | `<repo>/build/matlabc` | path to the matlabc binary |
| `WORKSPACE_ROOT` | `/tmp/matlab_llvm_workspaces` | per-session cwd jail root |
| `CPU_SECONDS` | `10` | RLIMIT_CPU per child |
| `MEMORY_MB` | `2048` | RLIMIT_AS per child (0 disables; see note) |
| `FILE_SIZE_MB` | `64` | RLIMIT_FSIZE per child |
| `MAX_PROCS` | `64` | RLIMIT_NPROC per child |
| `WALL_TIMEOUT_S` | `20` | hard wall-clock kill |
| `OUTPUT_CAP_BYTES` | `1000000` | max captured stdout/stderr |
| `MAX_UPLOAD_MB` | `25` | upload size cap |
| `API_TOKEN` | `""` | bearer token; empty disables auth |
| `SOURCE_CONTEXT_ROOT` | `<repo>` | root holding `docs/**/*.md` to index for RAG |
| `RAG_ENABLED` | `true` | build the doc index at startup |
| `RAG_TOP_K` | `4` | chunks retrieved per chat turn |
| `OPENAI_API_KEY` | `""` | enables the chat proxy (bare name; from `.env` or env) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | upstream chat endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | default model when the request omits one |

> **RLIMIT_AS caveat:** it counts *virtual* address space, which over-counts
> the large libLLVM/libMLIR mappings of a JIT process. Size it generously or
> set `0`; the container memory cgroup is the authoritative memory boundary.

## Layout

Flat module layout (run from `server/`, top-level imports):

```
config.py      settings (pydantic-settings)
sandbox.py     rlimit + timeout + cwd-jail + env-scrub launcher (run/spawn)
workspaces.py  per-user/session paths, traversal-safe resolve, artifact diff
matlabc.py     async wrappers around the real matlabc CLI
services.py    core ops shared by routers + MCP (check/repl/codegen/files)
diagnostics.py clang-style diagnostic parsing
rag.py         BM25 index over docs/**/*.md (dependency-free retrieval)
models.py      request/response schemas
auth.py        optional bearer-token dependency
mcp_tools.py   FastMCP server + tools (not named `mcp` — would shadow the SDK)
main.py        app factory + lifespan (+ MCP + RAG) + /healthz
routers/       check, repl, codegen, files, dap_ws, chat
tests/         pytest suite (fake matlabc stub in conftest.py)
```

## Tests

`just backend-test` (or `cd server && uv run --extra dev pytest`). The suite
uses a fake `matlabc` stub (`conftest.py`) so it runs anywhere — no compiler
build required. It covers the routes, diagnostics, figure capture, traversal
rejection, and the sandbox isolation guarantees (timeout, output cap, env
scrub).
