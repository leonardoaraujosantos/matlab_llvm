# Implementation Plan — matlab_llvm Remote Backend

> Companion to [`docs/remote_backend_trd.md`](remote_backend_trd.md).
> Target: a Coolify-hosted backend (FastAPI + FastMCP + the `matlabc`
> compiler) that powers a cross-platform mobile app (iPad / iPhone /
> Android) — "MATLAB in your pocket": program, compile, JIT-run (REPL),
> debug, plot, upload CSV/data, and download results.

This plan turns the corrected TRD into ordered, shippable phases. Every
`matlabc` invocation below is the real CLI verified in
`tools/matlabc/main.cpp`.

> **Implementation status (branch `feat_compiler_backend`).** The `server/`
> tree exists with **Phase 1** (`/v1/check`, `/v1/repl`, hardened sandbox),
> **Phase 2** (`/v1/codegen/*`), a **partial Phase 3** (workspaces +
> `/v1/files` + REPL figure capture), and **Phase 4** (`WS
> /v1/dap/ws/{session_id}` — opaque DAP-over-WebSocket byte bridge to
> `matlabc -dap`). **Phase 0/8** ship as the root `Dockerfile` +
> `docker-compose.yaml`. Run locally with `just backend-up`; test with `just
> backend-test` (fake matlabc, no LLVM build needed). See
> [`server/README.md`](../server/README.md). Deferred: Phase 5 (MCP),
> Phase 6 (chat/RAG), Phase 7 (auth/quotas/warm-pool), stateful sessions,
> and the dedicated `/v1/plot` route.

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
  mcp/
    server.py             # FastMCP tools -> mounted at /mcp/sse
  rag/
    indexer.py            # startup: chunk+embed docs/ + include/ + signatures
    store.py              # vector store (FAISS/Chroma) load/query
  tests/
    test_check.py test_repl.py test_codegen.py test_files.py
    test_dap_ws.py test_sandbox.py test_chat.py
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

## 7. Phase 5 — FastMCP / SSE

**Goal:** expose the compiler as MCP tools for AI clients.

* `mcp/server.py`: register tools `matlab_check`, `matlab_repl`,
  `matlab_codegen` (target enum), `matlab_plot`, `list_files`,
  `read_file` — thin wrappers over the Phase 1–3 services.
* Mount FastMCP's SSE app at `/mcp/sse` on the same FastAPI process/port.
* Reuse the same sandbox + workspace resolution so MCP and REST share
  isolation and per-user state.

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

* **Auth:** bearer token / API key middleware on every `/v1/*` and
  `/mcp/*` route; per-user identity drives workspace routing.
* **Sandboxing tier-2:** wrap children in `nsjail`/`firejail` (or run
  REPL/DAP in an ephemeral per-session container) for syscall-level
  isolation beyond `rlimit`; mount workspace read-write, everything else
  read-only; `--pids-limit`, `--cap-drop ALL`, `no-new-privileges`.
* **Quotas & rate limits:** per-user concurrent-job cap, request rate
  limit, disk quota, and a global worker semaphore so a burst can't
  exhaust the host.
* **Warm pool (perf, FIX-8):** maintain N pre-spawned `matlabc -repl`
  workers to amortize JIT startup for `/v1/repl`.
* **Observability:** structured logs, per-route latency/error metrics
  (Prometheus), and a `/healthz` for Coolify health checks.

---

## 10. Phase 8 — Coolify deploy & CI/CD

* Commit `Dockerfile` + `docker-compose.yaml` (TRD §5) at repo root (or
  under `deploy/`), point Coolify at the repo, map `OPENAI_API_KEY` +
  the resource-ceiling env vars, and attach the `workspace_data` volume.
* Traefik (Coolify) terminates TLS → HTTPS/WSS on `:443`; single
  upstream port `:8000`. Add the `/healthz` check and `restart: always`.
* CI (GitHub Actions): build the image, run the Phase-0 smoke gate +
  `server/tests/`, and on green let Coolify auto-deploy from `main`.
* Optional sidecar: `code-server` service for a full VS Code Web IDE
  reachable from Safari/Blink/Android browsers (deferred past v1).

---

## 11. Cross-cutting design decisions

* **Session model.** Two options:
  * *Stateless* — each `/v1/repl` is a fresh process (simple; no shared
    variables between calls).
  * *Stateful* — a long-lived `matlabc -repl` per session keeps the
    workspace variables across calls (matches desktop MATLAB; needs a
    session store, idle eviction, and the warm pool). **Recommended for
    the pocket-MATLAB UX**; start stateless in Phase 1, add stateful in
    Phase 7.
* **Mobile clients (iPad/iPhone/Android).** The backend is
  platform-neutral: REST + JSON for run/codegen/files, WebSocket for
  DAP, SSE for MCP and chat streaming. Native apps or a PWA render text
  output, inline figures (PNG/SVG), and a file browser; the DAP client
  can be a VS Code Web instance (code-server sidecar) or a custom thin
  DAP UI talking to `/v1/dap/ws`.
* **Artifacts.** Everything downloadable lives in the workspace volume;
  the API returns relative paths the client turns into
  `GET /v1/files/{path}` URLs. Consider short-lived signed URLs if the
  CDN/edge ever serves them directly.

---

## 12. Testing & acceptance (maps to TRD §6)

| Test | Phase | Pass condition |
|---|---|---|
| Build smoke | 0 | repl/codegen/plot run in the runtime image; `ldd` clean |
| Route payloads | 1–2 | `check`/`repl`/`codegen/*` return valid JSON from a mobile HTTP client |
| Isolation | 1 | crash, infinite loop, OOM each killed cleanly; FastAPI unaffected |
| Latency | 1 | `/v1/check` p95 `<200ms`; `/v1/repl` p95 within warm-pool target |
| Data round-trip | 3 | upload CSV → `readmatrix` → plot → download PNG |
| DAP stress | 4 | breakpoint/inspect/step/reverse over `/v1/dap/ws` |
| MCP tools | 5 | NL client triggers `matlab_repl`/`matlab_codegen` |
| Grounded chat | 6 | toolbox question retrieves correct doc chunk + cites it |

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
