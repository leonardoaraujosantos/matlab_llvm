# Implementation Plan — matlab_llvm Remote Backend

> Companion to [`docs/remote_backend_trd.md`](remote_backend_trd.md).
> Target: a Coolify-hosted backend (FastAPI + FastMCP + the `matlabc`
> compiler) that powers a cross-platform mobile app (iPad / iPhone /
> Android) — "MATLAB in your pocket": program, compile, JIT-run (REPL),
> debug, plot, upload CSV/data, and download results.

This plan turns the corrected TRD into ordered, shippable phases. Every
`matlabc` invocation below is the real CLI verified in
`tools/matlabc/main.cpp`.

> **Status: shipped + deployed 2026-05-25.** All phases 0–8 merged to `main`;
> the backend runs at <https://matlab-backend.coolify.cyberdynecorp.ai> on the
> Cyberdyne Coolify (project `MatlabLLVM` / env `production` / build pack
> `dockercompose`, tracks `main`). REST + DAP-over-WebSocket + **FastMCP at
> `/mcp/`** (streamable-HTTP — SSE is deprecated) + OpenAI-proxied
> `/v1/chat/completions` grounded in a dependency-free BM25 index over
> `docs/**/*.md` (proxies to OpenAI when `OPENAI_API_KEY` is set, else
> retrieval-only). Auth = **CyberdyneAuth bearer** validated against
> `${CYBERDYNE_AUTH_URL}/api/v1/users/me` on every `/v1/*`; MCP clients
> present an HMAC-signed bearer **minted by `POST /v1/mcp/token`** (30-day
> TTL, bound to the CyberdyneAuth identity). Stateful REPL (long-lived
> `matlabc -repl` per session, **15-min idle eviction**, warm pool size 2,
> **15 s wall-clock per turn**) + `/v1/plot` PNG/SVG/PDF + REPL figure capture.
> Tier-2 syscall sandbox plumbing (`bwrap`/`firejail`/`nsjail`) ships **but
> stays off on this host**: the lifespan-time `jail.probe()` found Docker's
> default seccomp/AppArmor profile blocks userns from inside the container
> even with `kernel.unprivileged_userns_clone=1` on the host; the probe
> downgrades automatically to `none` and surfaces the reason on `/healthz`
> under `sandbox.reason`. Real containment = container + non-root `runner`
> user + `rlimit` (CPU/AS/FSIZE/NPROC) + cwd jail (matlabc has no `cd`
> builtin) + scrubbed env + Docker default seccomp/AppArmor. Run locally
> with `just backend-up`; test with `just backend-test` (83 unit @ 92%, fake
> matlabc, no LLVM build), `just backend-itest` (live local uvicorn), or
> `just backend-test-remote URL '' USER PASS [AUTH_URL] [EXPECT_SANDBOX]`
> (same 21-test suite against any deploy). See
> [`server/README.md`](../server/README.md).

---

## 0. Guiding constraints (from the codebase review)

These shape every phase; see TRD §7 for the evidence.

1. **`matlabc` is the DAP adapter** (`matlabc -dap <prog.m>`), speaking
   DAP over **stdio** with `Content-Length` framing. There is no
   `lldb-dap` and no DAP socket — the network edge is a stdio↔WebSocket
   bridge we write in FastAPI.
2. **REPL is text + stateful over stdin** (`matlabc -repl`); the default
   mode (run with **no flag** — there is no `-check` flag) is a validate-only
   check (no execution). JSON is the API layer's job.
3. **`matlabc` dynamically links libLLVM/libMLIR** and (with plotting)
   Cairo — the runtime image must carry those `.so`s.
4. **LLVM version must match the source** (20 stable floor; 22 = current
   dev). Install prebuilt LLVM+MLIR from `apt.llvm.org`; never build LLVM
   from source.
5. **REPL/DAP run arbitrary native code** — treat every child process as
   untrusted: timeout + `rlimit` + non-root + confined cwd + (ideally)
   syscall sandbox.

---

## 1. Repository layout

Add a `server/` tree (Python) alongside the existing C++ project. Keep
the backend self-contained so the Dockerfile `COPY server/` is clean.

```
server/
  pyproject.toml | requirements.txt
  main.py                  # FastAPI app factory, router mounting, lifespan
  config.py                # env-driven settings (pydantic-settings)
  matlabc.py               # thin async wrapper around the matlabc binary
  sandbox.py               # rlimit/timeout/cwd-jail process launcher
  workspaces.py            # per-user/session workspace paths + quotas
  routers/
    check.py               # POST /v1/check
    repl.py                # POST /v1/repl   (+ figure capture)
    codegen.py             # POST /v1/codegen/{python,typescript,c,cpp,systemverilog}
    files.py               # POST/GET /v1/files  (upload/list/download)
    plot.py                # POST /v1/plot  (snippet -> figure bytes)
    chat.py                # POST /v1/chat/completions (OpenAI-compatible)
    dap_ws.py              # WS /v1/dap/ws/{session_id}  (stdio bridge)
  mcp_tools.py             # FastMCP tools -> mounted at /mcp/ (streamable-HTTP)
                           # NB: file is mcp_tools.py, NOT a `mcp/` package —
                           # a local mcp/ would shadow the installed mcp SDK.
  mcp_auth.py              # HMAC mint/verify for backend-issued MCP tokens
  rag.py                   # dependency-free BM25 over docs/**/*.md
  auth.py                  # CyberdyneAuth bearer validation + cache
  jail.py                  # tier-2 sandbox argv builders + startup probe
  sessions.py              # stateful REPL session manager + warm pool
  limits.py                # rate limit / global concurrency / disk quota
  tests/                   # unit tests (fake matlabc, no LLVM build needed)
  integration/             # live-server suite — local boot or remote URL
Dockerfile                 # multi-stage (see TRD §5)
docker-compose.yaml        # Coolify (see TRD §5)
```

`requirements.txt` (starting set): `fastapi`, `uvicorn[standard]`,
`pydantic-settings`, `python-multipart`, `websockets`, `sse-starlette`,
`fastmcp`, `openai`, `httpx`, and for RAG `faiss-cpu` (or `chromadb`) +
an embeddings client.

---

## 2. Phase 0 — Build & containerize `matlabc`

**Goal:** a reproducible Linux build and a runtime image that boots.

* Resolve the LLVM/MLIR version the source compiles against (try 20
  first; bump to 21/22 if `find_package`/compile fails). Record it as
  the `LLVM_VERSION` build arg.
* Author the multi-stage `Dockerfile` per TRD §5: builder installs
  `apt.llvm.org` LLVM+MLIR dev + `libcairo2-dev` and compiles `matlabc`
  with `-DMATLAB_LLVM_WITH_PLOT=ON`; runtime installs the matching
  `libllvm`/`libmlir`/`libcairo2` runtime libs and copies the binary +
  `source_context`.
* **Smoke gate:** in the runtime image, `matlabc -repl <<< "disp(1+1)"`
  prints `2`, `echo "x=[1 2;3 4]; disp(x)" | matlabc -emit-python -`
  emits Python, and a `saveas(gcf,'/tmp/a.png')` snippet writes a PNG.
* Confirm dynamic-lib resolution with `ldd /usr/local/bin/matlabc`
  showing no "not found".

**Exit criteria:** image builds in CI and the three smoke commands pass.

---

## 3. Phase 1 — FastAPI core: `/v1/check` + `/v1/repl` + sandbox

**Goal:** the two highest-value endpoints, on a hardened launcher.

* `sandbox.py`: an async subprocess launcher applying, per child —
  `preexec_fn` with `setrlimit` (CPU seconds, address space ≈
  `REPL_MEMORY_MB`, file size, `RLIMIT_NPROC`), a hard wall-clock
  `asyncio.wait_for` timeout that kills the process group, cwd set to the
  session workspace, and a scrubbed env. Capture stdout/stderr with a
  byte cap.
* `matlabc.py`: helpers `check(src)`, `repl(src)`, `emit(target, src)`
  that write `src` to a temp `.m` in the workspace (or feed `-repl` over
  stdin) and shell out via `sandbox.py`.
* `POST /v1/check` → `matlabc file.m` (Check is the **default** mode; there
  is no `-check` flag — passing one errors as "unknown flag"); return
  `{ok, diagnostics[], stderr}`. **This is the only `<200ms` route.**
* `POST /v1/repl` → drive `matlabc -repl`: write code to stdin, read
  stdout/stderr, return `{stdout, stderr, ok, artifacts[]}` (artifacts
  filled in Phase 3). Decide session model in §11.

**Exit criteria:** unit tests for normal output, syntax error,
infinite-loop timeout, and OOM kill (the isolation acceptance test).

---

## 4. Phase 2 — Codegen routes

**Goal:** transpilation endpoints, one per real flag.

* `POST /v1/codegen/python` → `-emit-python`
* `POST /v1/codegen/typescript` → `-emit-typescript`
* `POST /v1/codegen/c` → `-emit-c`
* `POST /v1/codegen/cpp` → `-emit-cpp`
* `POST /v1/codegen/systemverilog` → `-emit-systemverilog`

Each: write snippet to a temp `.m`, run the flag, return
`{code, language, ok, diagnostics}`. Optionally persist the output as a
downloadable artifact (Phase 3). Reuse the Phase-1 sandbox (codegen is
pure compilation — cheap, but still bounded).

**Exit criteria:** golden tests asserting each target emits non-empty,
language-appropriate output for a shared sample `.m`.

---

## 5. Phase 3 — Pocket-MATLAB I/O: workspaces, files, plots (TRD F5)

**Goal:** the data-in / results-out loop that makes it feel like MATLAB.

* `workspaces.py`: resolve `/workspace/{user_id}/{session_id}/`, create
  on demand, enforce per-user disk quota, and hand the path to the
  sandbox as cwd.
* **Upload:** `POST /v1/files` (multipart) → store into the workspace
  (validate extension/size; reject path traversal). User code then reads
  with `readmatrix('data.csv')` / `readtable(...)`.
* **List/Download:** `GET /v1/files` (tree) and `GET /v1/files/{path}`
  (stream bytes, content-type by extension). Covers uploaded data,
  saved codegen output, `save`'d `.mat`, `writematrix` CSV, and figures.
* **Plot route:** `POST /v1/plot` runs a snippet that calls `plot(...)`
  + `saveas(gcf, OUT)` (server injects `OUT` in the workspace) and
  streams the resulting PNG/SVG/PDF; format chosen by an Accept header
  or `?format=`. (In-memory `matlab_render_png` is an optimization for
  later; disk `saveas` is the simplest first cut.)
* **REPL figure capture:** after a `/v1/repl` turn, diff the workspace
  for new image files and return them as `artifacts[]` (download URLs)
  in the REPL response.

**Exit criteria:** upload a CSV → `readmatrix` it in `/v1/repl` →
produce a plot → download the PNG, all in one integration test.

---

## 6. Phase 4 — DAP-over-WebSocket bridge

**Goal:** remote step-debugging for the mobile app's editor.

* `WS /v1/dap/ws/{session_id}`: on connect, spawn
  `matlabc -dap <workspace/program.m>` under the sandbox; run two pumps —
  WebSocket→child.stdin and child.stdout→WebSocket — preserving the DAP
  `Content-Length` framing as an opaque byte stream (the client owns DAP
  semantics).
* Lifecycle: one child per WS connection; kill the child (and its group)
  on disconnect/timeout; surface child stderr as a diagnostic frame.
* Because the DAP server JIT-executes the program **in-process**, the
  same sandbox limits apply; a crash dies with the child, never the
  bridge.
* Validate against the in-repo client `test/Debug/dap_client.py`
  re-pointed through the WebSocket (set breakpoint → stop → `variables`
  → `next` → `continue`).

**Exit criteria:** a scripted DAP session over the WebSocket reproduces
the breakpoint/inspect/step flow from `test/Debug/`.

---

## 7. Phase 5 — FastMCP (streamable-HTTP at `/mcp/`)

**Goal:** expose the compiler as MCP tools for AI clients.

* `mcp_tools.py`: register tools `matlab_check`, `matlab_repl`,
  `matlab_codegen` (target enum), `list_files`, `read_file` — thin
  wrappers over the Phase 1–3 services (the dedicated `matlab_plot`
  tool is folded into `matlab_repl` + figure-capture artifacts).
* Mount FastMCP's streamable-HTTP app at `/mcp/` on the same FastAPI
  process/port. (The plan originally proposed `/mcp/sse`; SSE was
  deprecated by MCP — we ship modern streamable-HTTP. **Note:** the
  module is `mcp_tools.py`, NOT a `mcp/` package — a local `mcp/`
  would shadow the installed `mcp` SDK that FastMCP imports.)
* Reuse the same sandbox + workspace resolution so MCP and REST share
  isolation and per-user state. MCP auth is handled in Phase 7
  (backend-minted HMAC tokens, not CyberdyneAuth bearers directly —
  MCP clients can't run the OAuth/refresh flow).

**Exit criteria:** an MCP client (e.g. `mcp-client` skill) lists the
tools and successfully invokes `matlab_repl` and `matlab_codegen`.

---

## 8. Phase 6 — OpenAI-compatible chat + retrieval RAG

**Goal:** `/v1/chat/completions` grounded in the compiler's own docs.

* `rag/indexer.py` (startup/lifespan): chunk + embed the **high-signal**
  corpus — `docs/*.md` roadmaps, `include/` public headers, and an
  extracted builtin/function **signature index** — into a vector store.
  *Do not* stuff raw `.cpp`/`.h` per request (TRD §7 FIX-7).
* `routers/chat.py`: accept the OpenAI schema; per request, retrieve
  top-k chunks for the user's message and prepend a compact, cited
  context block to the `system_prompt`; proxy to OpenAI (or a local
  model) with streaming (`stream: true`) passthrough.
* Scope: architecture Q&A, decoding LLVM/MLIR errors, autocompleting
  valid MATLAB-subset blocks.

**Exit criteria:** a question about, say, `fmincon` or `-emit-sv`
retrieves the right roadmap chunk and answers with a citation; the
endpoint is drop-in for an OpenAI Base URL override.

---

## 9. Phase 7 — Hardening, auth, observability

* **Auth (resolved as CyberdyneAuth + minted MCP token):** `auth_mode`
  is picked at startup from env:
  * `cyberdyne` when `CYBERDYNE_AUTH_URL` is set — every `/v1/*` bearer
    is validated against `${CYBERDYNE_AUTH_URL}/api/v1/users/me`
    (`auth.py`). Successful checks are cached for
    `AUTH_VERIFY_CACHE_TTL_S`. The verified id becomes the request
    *principal*, so workspaces are isolated per CyberdyneAuth UUID
    regardless of any client-sent `user_id`.
  * `token` when `MATLAB_BACKEND_API_TOKEN` is set — single shared static
    bearer (handy for CI / curl probes).
  * `none` otherwise — open (local-dev default).
  **MCP** uses a separate flow: an authenticated REST caller mints a
  stateless **HMAC-signed bearer** via `POST /v1/mcp/token`
  (`mcp_auth.py`, 30-day TTL, payload binds the CyberdyneAuth `sub`).
  MCP clients present that bearer on `/mcp/`; revocation = rotate
  `MATLAB_BACKEND_MCP_TOKEN_SECRET` on the host.
* **Sandboxing tier-2:** `jail.py` plumbing for `bwrap`/`firejail`/
  `nsjail`. **Stays off in production**: a lifespan-time
  `jail.probe()` runs the wrapper on a no-op argv before declaring it
  active; on failure `settings.sandbox_backend` is downgraded to
  `none` and the reason surfaces on `/healthz` under `sandbox.reason`.
  On the Cyberdyne Coolify host, Docker's default seccomp + AppArmor
  profile blocks userns creation from inside the container even with
  `kernel.unprivileged_userns_clone=1` on the host, so bwrap can't run
  without weakening Docker's own MAC profile — explicitly *not* done,
  since the container's namespaces + non-root + rlimit + cwd jail +
  Docker default seccomp/AppArmor already give layered isolation that
  bwrap would mostly duplicate.
* **Quotas & rate limits (`limits.py`):** per-client sliding-window
  rate limit (`rate_limit_per_minute`, default 120; 429 on overflow),
  global concurrency semaphore on matlabc children
  (`max_concurrent_jobs`, default 8 — excess queues), per-workspace
  disk quota (`user_quota_mb`, default 200). **Open follow-on:** no
  per-user *session count* cap yet — users can mint arbitrary
  `session_id`s (memory is the practical bound, with idle eviction).
* **Stateful REPL + warm pool (`sessions.py`):** long-lived
  `matlabc -repl` per `(user_id, session_id)`, each turn delimited by
  a stdin-injected marker (`disp('<<<MLBC_TURN_uuid>>>')`). Idle
  eviction at `repl_idle_timeout_s` (15 min) via background sweep
  every `repl_evict_interval_s` (60 s). `warm_pool_size` workers
  (default 2) are pre-JIT-warmed at lifespan startup; the first N
  sessions adopt a pool worker's cwd (matlabc has no `cd` builtin,
  so the worker can't be retargeted — files are migrated on adopt /
  retire).
* **Observability:** `/healthz` returns the matlabc path + tier-2
  sandbox state (`{backend, active, allow_net, reason?}`). Structured
  logs + per-route latency/error metrics (Prometheus) are an open
  follow-on.

---

## 10. Phase 8 — Coolify deploy & CI/CD

* **Deployed.** App `matlab-llvm-backend` (uuid `go4s0sw0oo048gs0kc48ogoo`)
  on `https://coolify.cyberdynecorp.ai`, project `MatlabLLVM`, env
  `production`. Source: this repo on branch `main`, build pack
  `dockercompose`, compose file `/docker-compose.yaml`, expose `:8000`,
  FQDN `https://matlab-backend.coolify.cyberdynecorp.ai` (Traefik
  terminates TLS → HTTPS/WSS). Persistent workspace volume mounted at
  `/workspace`. Coolify env vars:
  `MATLAB_BACKEND_MCP_REQUIRE_AUTH=1`, `MATLAB_BACKEND_MCP_TOKEN_SECRET`
  (64-hex), `OPENAI_API_KEY` + `OPENAI_BASE_URL` (from prod
  geo_dashboard — the local `.env` is stale), `CYBERDYNE_AUTH_URL`
  (also defaults in compose to the same value). `MATLAB_BACKEND_SANDBOX_BACKEND`
  is **not set** (compose default `none`) — see Phase 7.
* **CI (`.github/workflows/backend.yml`):** `tests` job is fast, runs
  on any backend-related push (server/+Dockerfile+compose), gates on
  `--cov-fail-under=90` + live integration. The heavier `image` job
  (build + smoke gate on the real container) is **opt-in via the
  `backend-image` PR label or `workflow_dispatch`** to avoid the
  LLVM build on every push; flip to auto-on-`main` once the host
  proves stable.
* **Optional sidecar:** `code-server` for a full VS Code Web IDE on
  mobile browsers — explicitly deferred past v1.

---

## 11. Cross-cutting design decisions

* **Session model — resolved as stateful.** A long-lived `matlabc -repl`
  per `(user_id, session_id)` keeps workspace variables across `/v1/repl`
  turns (matches desktop MATLAB UX). Backed by `sessions.SessionManager`
  + warm pool + idle eviction (Phase 7). Per-request override available
  via `stateful: false` for one-shot semantics.
* **Mobile clients (iPad/iPhone/Android).** The backend is
  platform-neutral: REST + JSON for check/run/codegen/files,
  **WebSocket** for DAP, **streamable-HTTP** for MCP and **SSE** for
  chat streaming. Native apps or a PWA render text output, inline
  figures (PNG/SVG), and a file browser; the DAP client can be a VS
  Code Web instance (code-server sidecar, deferred) or a custom thin
  DAP UI talking to `/v1/dap/ws`. *No mobile client has been written
  yet — this is the largest remaining gap to end-user value.*
* **Artifacts.** Everything downloadable lives in the workspace volume;
  the API returns relative paths the client turns into
  `GET /v1/files/{path}` URLs. Short-lived signed URLs (for direct
  CDN/edge serving) are an open follow-on.

---

## 12. Testing & acceptance (maps to TRD §6)

| Test | Phase | Pass condition | Status |
|---|---|---|---|
| Build smoke | 0 | `matlabc` repl/codegen/plot run in the runtime image | ✅ |
| Route payloads | 1–2 | `check`/`repl`/`codegen/*` return valid JSON | ✅ 5 codegen targets covered |
| Isolation | 1 | crash, infinite loop, OOM each killed cleanly; FastAPI unaffected | ✅ unit + live |
| Latency | 1 | `/v1/check` p95 `<200ms`; `/v1/repl` p95 within warm-pool target | ✅ observed |
| Data round-trip | 3 | upload CSV → `readmatrix` → plot → download PNG | ✅ |
| DAP stress | 4 | breakpoint/inspect/step/reverse over `/v1/dap/ws` | ✅ initialize/launch/configurationDone/threads + events verified |
| MCP tools | 5 | NL client triggers `matlab_check`/`matlab_repl`/`matlab_codegen` via the **`mcp-client` skill** | ✅ |
| MCP auth | 7 | unauthenticated `/mcp/` rejected; minted token accepted | ✅ |
| Grounded chat | 6 | toolbox question retrieves doc chunks + cites; tolerates LLM 4xx | ✅ live with `gpt-4o-mini` |
| Sandbox state | 7 | `/healthz` reflects runtime backend + reason on probe failure | ✅ |

---

## 13. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| LLVM/MLIR version mismatch on `apt.llvm.org` (no MLIR dev pkg for chosen v) | Med | Pin to a version with `libmlir-<v>-dev`; else copy `.so`s from builder or static-link |
| Arbitrary-code execution abuse | High | Layered sandbox (rlimit + nsjail/container), quotas, auth — Phase 7 is not optional for public exposure |
| JIT cold-start latency on mobile networks | Med | Warm pool; keep WS sessions alive; scope `<200ms` to the no-flag check only |
| Image bloat from LLVM runtime libs | Med | Slim runtime stage; consider static link or stripped `.so`s |
| Stateful REPL memory growth / leaks per session | Med | Idle eviction, per-session memory cap, periodic worker recycle |
| RAG context cost/quality | Low | Index docs+signatures (not raw source); cap top-k; cache embeddings |

---

## 14. Recommended build order (shortest path to a usable demo)

1. **Phase 0** (image boots) → **Phase 1** (`/v1/check` + `/v1/repl`).
2. **Phase 3** (files + plots) — this is what makes the mobile demo
   compelling: upload CSV, run, see a plot, download it.
3. **Phase 2** (codegen) — cheap once Phase 1 exists.
4. **Phase 4** (DAP-over-WS) — the differentiator vs. a plain REPL box.
5. **Phase 5/6** (MCP + chat) — intelligence layer.
6. **Phase 7/8** (harden + Coolify) — required before any public/shared
   exposure.

> First milestone to show on an iPad: Phases 0 → 1 → 3 — type MATLAB,
> upload a CSV, get a plot back, download the result.
