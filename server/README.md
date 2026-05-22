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
- **Phase 7** — hardening: auth on all `/v1` routes (CyberdyneAuth bearer
  validation, or a static token), per-identity workspace isolation, per-client rate
  limiting, a global concurrency cap on matlabc children, per-workspace disk
  quotas, **stateful REPL sessions** (long-lived `matlabc -repl` per session,
  idle-evicted), a **warm pool** of pre-warmed workers, and a **tier-2 syscall
  sandbox** (bubblewrap/firejail/nsjail).
- **`/v1/plot`** — run a plotting snippet and stream back PNG/SVG/PDF.
- **Phase 0/8** — `Dockerfile` + `docker-compose.yaml` at the repo root.

Deferred: only the CI/deploy half of Phase 8.

> **Warm pool note:** matlabc has no `cd` builtin, so a pre-spawned worker
> can't be retargeted to a session's deterministic dir. Instead a pooled
> worker *adopts its own pool dir as the session workspace* — staged files
> migrate in on adoption and back out on retirement. Set `WARM_POOL_SIZE=0`
> to disable.

> **Tier-2 sandbox note:** Linux-only; falls back to rlimit-only if the tool
> is missing (e.g. on macOS dev). The container is the primary boundary; this
> is defense-in-depth. Enable with `SANDBOX_BACKEND=bwrap`.

> The plan named the MCP mount `/mcp/sse`; SSE transport is deprecated in
> MCP, so this ships modern **streamable-HTTP** at `/mcp` instead.

## Secrets / `.env`

Secrets load from the environment or a **gitignored** `server/.env`:
- `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) for the chat
  proxy; without it `/v1/chat/completions` runs in retrieval-only mode.
- `CYBERDYNE_AUTH_URL` to enable token auth (see below).

## Auth

`auth_mode` is resolved automatically:

| Mode | When | Behaviour |
|---|---|---|
| `cyberdyne` | `CYBERDYNE_AUTH_URL` set | Validates each `Authorization: Bearer <token>` against CyberdyneAuth's `GET /api/v1/users/me` (200 = valid; 401/403 = reject; 5xx = `503`). Successful checks are cached for `AUTH_VERIFY_CACHE_TTL_S`. The verified identity id becomes the request **principal**, so each user's workspace is isolated regardless of the `user_id` they send. |
| `token` | `MATLAB_BACKEND_API_TOKEN` set | A single shared static bearer token. |
| `none` | neither | Open (local-dev default). |

`GET /v1/auth/whoami` echoes the authenticated identity. The DAP WebSocket
authenticates via a `?token=` query param (browsers can't set `Authorization`
on a WS handshake). MCP (`/mcp`) is not yet behind auth — a follow-on.

## Run it locally

From the repo root (needs [`just`](https://github.com/casey/just) and
[`uv`](https://docs.astral.sh/uv/)):

```sh
just backend-up          # builds matlabc (+ Cairo plotting if present), serves :8000
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
| GET  | `/v1/auth/whoami` | echo the authenticated identity (auth smoke) |
| POST | `/v1/check` | validate-only (default matlabc mode) |
| POST | `/v1/repl` | JIT-execute; returns stdout/stderr + figure artifacts |
| POST | `/v1/codegen/{target}` | transpile to python/typescript/c/cpp/systemverilog |
| POST | `/v1/files` | multipart upload into the session workspace |
| GET  | `/v1/files` | list the workspace tree |
| GET  | `/v1/files/{path}` | download a file |
| WS   | `/v1/dap/ws/{session_id}` | DAP-over-WebSocket bridge (`?program=`, `?token=`) |
| MCP  | `/mcp` | FastMCP tools over streamable-HTTP for AI clients |
| POST | `/v1/chat/completions` | OpenAI-compatible chat grounded in the docs (RAG) |
| POST | `/v1/plot` | run a plotting snippet, stream PNG/SVG/PDF (`?format=`) |

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
| `CYBERDYNE_AUTH_URL` | `""` | enable CyberdyneAuth bearer validation (bare name) |
| `API_TOKEN` | `""` | shared static bearer token (used if no CyberdyneAuth) |
| `AUTH_VERIFY_CACHE_TTL_S` | `30` | cache window for `/users/me` checks |
| `MAX_CONCURRENT_JOBS` | `8` | global cap on concurrent matlabc children |
| `RATE_LIMIT_PER_MINUTE` | `120` | per-client request cap (0 disables) |
| `USER_QUOTA_MB` | `200` | per-workspace disk quota (0 disables) |
| `REPL_STATEFUL` | `true` | keep a long-lived `matlabc -repl` per session |
| `REPL_IDLE_TIMEOUT_S` | `900` | evict sessions idle longer than this |
| `WARM_POOL_SIZE` | `2` | pre-warmed `matlabc -repl` workers (0 disables) |
| `SANDBOX_BACKEND` | `none` | tier-2 jail: `none`/`bwrap`/`firejail`/`nsjail` |
| `SANDBOX_ALLOW_NET` | `false` | give jailed children network access |
| `PLOT_DEFAULT_FORMAT` | `png` | default `/v1/plot` format |
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
sessions.py    stateful REPL session manager + warm pool (adoption/migration)
limits.py      concurrency cap + rate limiter + disk-quota helper
jail.py        tier-2 syscall sandbox wrapper (bwrap/firejail/nsjail)
diagnostics.py clang-style diagnostic parsing
rag.py         BM25 index over docs/**/*.md (dependency-free retrieval)
models.py      request/response schemas
auth.py        auth modes (cyberdyne / static token / open) + verification
principal.py   request-scoped identity → per-user workspace isolation
mcp_tools.py   FastMCP server + tools (not named `mcp` — would shadow the SDK)
main.py        app factory + lifespan (+ MCP + RAG + warm pool) + /healthz
routers/       check, repl, codegen, files, dap_ws, chat, plot, whoami
tests/         pytest suite (fake matlabc stub in conftest.py)
```

## Tests

`just backend-test` (or `cd server && uv run --extra dev pytest`). The suite
uses a fake `matlabc` stub (`conftest.py`) so it runs anywhere — no compiler
build required. It covers the routes, diagnostics, figure capture, traversal
rejection, the DAP bridge, MCP tools, RAG + chat (offline and mocked-proxy),
stateful-session persistence/eviction, the hardening limits, and the sandbox
isolation guarantees (timeout, output cap, env scrub).

`just backend-cov` runs the same suite with a coverage report — **92%**
(the gaps are the `preexec` child, which runs post-fork and can't be
instrumented, plus a few defensive error branches).

`just backend-itest` runs **live-server integration tests** (`integration/`):
it boots a real `uvicorn` subprocess and drives it over real HTTP/WS —
check, stateful REPL, codegen, file round-trip, plot, chat, MCP-over-HTTP,
and the DAP WebSocket bridge. It uses `build/matlabc` when present (the real
compiler), otherwise the fake stub, so it runs anywhere. (The plot test skips
if the real build lacks `-DMATLAB_LLVM_WITH_PLOT`.)
