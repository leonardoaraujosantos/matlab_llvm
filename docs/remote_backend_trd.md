# Technical Requirements Document (TRD)
## Project: matlab_llvm Remote Backend ("MATLAB in your pocket")

> **Status:** reviewed for feasibility against the actual codebase and
> corrected. See [§7 Feasibility Assessment & Corrections](#7-feasibility-assessment--corrections)
> for every change made to the original draft and the evidence behind it.
> The companion implementation plan is in
> [`docs/remote_backend_plan.md`](remote_backend_plan.md).

---

## 1. Product Overview

The matlab_llvm Remote Backend is a remotely-hosted development
infrastructure that exposes the capabilities of the
[matlab_llvm](https://github.com/leonardoaraujosantos/matlab_llvm)
compiler as a network service. It removes the native-execution
limitation of mobile OSes and powers a cross-platform app — **iPad,
iPhone, and Android** — that lets users **program, compile, JIT-run
(REPL), debug, plot, upload data, and download results** as if they
carried a MATLAB server in their pocket. Automated deployment is managed
through **Coolify**, enabling continuous delivery (CI/CD) straight from
the Git repository.

### Primary Goals

* **Full portability** — author MATLAB, run it through the JIT REPL,
  generate code (Python / TypeScript / C / C++ / SystemVerilog), produce
  plots, and step-debug it, from any mobile device (iPadOS / iOS /
  Android) over the network.
* **Data in, results out** — upload CSV/data files for analysis, render
  plots as PNG/SVG/PDF, and download generated code, figures, and
  computed result files.
* **Integrated intelligence** — an OpenAI-compatible chat endpoint with
  deep, retrieval-grounded context about the compiler's internals and
  its shipped toolboxes.
* **Unified management** — consolidate the communication channels
  (REST, MCP/SSE, WebSocket) and the debug stack (DAP) behind a single
  Python orchestrator (FastAPI) in a Coolify-managed container.

---

## 2. System Architecture & Data Flow

The original draft proposed three peer containers, one of them a
dedicated `lldb-dap` service. **That is incorrect for this project:
`matlabc` *is its own* Debug Adapter and speaks DAP over stdio — there
is no `lldb-dap` and no DAP network port** (see §7, FIX-1). The
corrected architecture is a single application container that owns the
`matlabc` binary and brokers every channel; the DAP transport is a
per-session stdio↔WebSocket bridge inside FastAPI. `code-server` stays
an optional sidecar.

```
+----------------------------------------------------------------------------+
|                              COOLIFY STACK                                  |
|                                                                            |
|  [ Traefik / Coolify reverse proxy ]  — TLS termination + routing          |
|             |                                          |                   |
|     :443  HTTPS / WSS                          :443 (optional path)         |
|     (REST + MCP/SSE + DAP-over-WS)             (code-server IDE)            |
|             v                                          v                    |
|  +-------------------------------------+   +-----------------------------+  |
|  | Container 1: matlab-llvm-backend    |   | Container 2 (optional):     |  |
|  |   FastAPI + FastMCP  (orchestrator) |   |   code-server (VS Code Web) |  |
|  |                                     |   |   reached from Safari /      |  |
|  |   - REST: /v1/repl, /v1/codegen/*   |   |   Blink Shell on the iPad    |  |
|  |   - /v1/chat/completions (OpenAI)   |   +-----------------------------+  |
|  |   - MCP/SSE: /mcp/sse               |                                    |
|  |   - DAP bridge: /v1/dap/ws/{sid}    |  spawns, per session:              |
|  |        |                            |    `matlabc -dap <prog.m>`         |
|  |        +--- stdio <--> WebSocket ---+----> (native DAP over stdin/stdout)|
|  |   - bundles the `matlabc` binary    |                                    |
|  |     + libLLVM/libMLIR runtime libs  |                                    |
|  +-------------------------------------+                                    |
|             |                                                               |
|             v   subprocess (rlimit + timeout + sandbox)                     |
|     `matlabc -repl`  /  `matlabc -emit-*`  /  `matlabc -check`              |
+----------------------------------------------------------------------------+
                                  |
                                  v
                        persistent volume  /workspace
```

**Key architectural facts (verified against the source):**

| Concern | Reality in `matlab_llvm` |
|---|---|
| DAP adapter | `matlabc -dap <program.m>` — the binary itself is the adapter. No `lldb-dap`. |
| DAP transport | **stdio** with LSP-style `Content-Length` JSON framing. No TCP/socket code exists anywhere in `lib/` or `tools/`. |
| DAP execution | The paused program is **JIT-executed in-process** (MLIR `ExecutionEngine`). A crash takes down *that* subprocess only. |
| REPL | `matlabc -repl` — stateful JIT REPL reading stdin line/block at a time; emits **plain MATLAB-style text**, not JSON. |
| Default mode | No flag ⇒ `-check` (lex + parse + Sema only, **does not execute**). There is no "run file and exit" mode; use `-repl`. |
| Codegen | `-emit-python`, `-emit-typescript`/`-emit-ts`, `-emit-c`, `-emit-cpp`, `-emit-systemverilog`/`-emit-sv` (plus `-emit-llvm`, `-emit-mlir`, `-emit-cocotb`, `-emit-matlab`, `-emit-mflow`). Output is text on stdout. |
| Plotting | Cairo backend behind `-DMATLAB_LLVM_WITH_PLOT=ON`. `plot`/`title`/… → `saveas`/`print` to **PNG/SVG/PDF** on disk, **and** an in-memory C-ABI (`matlab_render_png/svg/pdf` → malloc'd buffer). Headless: no display server. Needs `libcairo2-dev`. |
| File I/O | `readtable`/`readmatrix` (CSV/delimited, auto-delimiter + per-column type inference), `writematrix`/`writetable`, `fopen`/`fread`/`fwrite`, `load`/`save`. User code reads uploaded files by path. |
| LLVM/MLIR | Built against prebuilt LLVM+MLIR via `find_package(LLVM/MLIR CONFIG)`. Dev env: Homebrew **llvm@22** (libLLVM 22.1.3). `matlabc` **dynamically links** `libLLVM`/`libMLIR`. |

---

## 3. Functional Specifications (Features)

### F1. Docker Infrastructure & Coolify Integration

* **Automated build (multi-stage).** Stage 1 (`builder`) **installs**
  prebuilt LLVM + MLIR dev packages (`llvm-<v>-dev`, `libmlir-<v>-dev`,
  `mlir-<v>-tools`, `clang-<v>`, `lld-<v>`) from the official LLVM apt
  repository (`apt.llvm.org`), plus `cmake`, `ninja-build`, `git`, then
  compiles **only** the `matlab_llvm` project. **It does not build LLVM
  from source** (that is a multi-hour, 16 GB+ job — see §7, FIX-3).
  Stage 2 (`runtime`) is a lean image carrying the `matlabc` binary,
  the **matching `libLLVM`/`libMLIR` runtime shared libraries** (FIX-4),
  and Python 3.12.
* **Coolify support (Docker Compose).** A `docker-compose.yaml` with
  env-var mapping (`OPENAI_API_KEY`) and a persistent volume for user
  source under `/workspace`.

### F2. Execution & Debug Engine (JIT REPL + DAP)

* **JIT REPL mode.** `POST /v1/repl` drives `matlabc -repl`: FastAPI
  writes the posted MATLAB code to the REPL subprocess's **stdin**,
  reads its **stdout/stderr**, and wraps the captured buffers in JSON.
  *The JSON envelope is produced by FastAPI — `matlabc` emits text.*
  Optionally keep a warm pool of REPL workers to amortize JIT startup.
* **Native DAP bridge (not a standalone port).** Debugging is reached
  **only** through the WebSocket bridge in F3: per session, FastAPI
  spawns `matlabc -dap <prog.m>` and pumps bytes between the WebSocket
  and the subprocess's stdin/stdout. There is **no** independent DAP
  TCP listener and **no** `lldb-dap`.
* **Debug capabilities (already implemented in `matlabc -dap`).**
  breakpoints (incl. conditional + logpoints), `scopes`/`variables`
  inspection, `next`/`stepIn`/`stepOut`, `stepBack`/`reverseContinue`,
  matrix `readMemory`/`writeMemory`, and `evaluate` (REPL against the
  paused session). Consumed from a DAP-speaking client on the iPad
  (e.g. VS Code Web).

### F3. Hybrid Communication Layer

* **REST interface (FastAPI):**
  * `POST /v1/check` — fast syntax/semantic validation (`matlabc -check`).
  * `POST /v1/repl` — immediate JIT execution.
  * `POST /v1/codegen/python` — transpile to NumPy-structured Python (`-emit-python`).
  * `POST /v1/codegen/typescript` — transpile to TypeScript (`-emit-typescript`).
  * `POST /v1/codegen/cpp` — transpile to C++ (`-emit-cpp`).
  * `POST /v1/codegen/c` — transpile to C (`-emit-c`).
  * `POST /v1/codegen/systemverilog` — synthesizable RTL (`-emit-systemverilog`).
    *(Original draft folded Python+TypeScript into one route and called
    Verilog "verilog"; these are distinct flags — see §7, FIX-6.)*
* **FastMCP interface (Model Context Protocol):** an SSE server at
  `/mcp/sse` that auto-publishes the compiler operations (check, repl,
  each codegen target) as MCP tools callable by MCP-aware AI clients on
  the iPad.
* **Low-latency WebSocket channel:** `/v1/dap/ws/{session_id}` — the DAP
  stdio↔WebSocket proxy. This is **mandatory** (it is the only remote
  path to the debugger), not the optional convenience the original draft
  described (FIX-1). It also stabilizes long-lived sessions over mobile
  4G/5G by keeping DAP inside an HTTP-upgraded WebSocket.

### F4. OpenAI-compatible Agent (internal RAG)

* **API compatibility.** `POST /v1/chat/completions` mirroring the
  OpenAI schema, so any iPadOS third-party app can attach by setting the
  Base URL.
* **Code-context injection (retrieval RAG, not full-file stuffing).**
  At startup an indexer chunks and embeds the *high-signal* corpus —
  the `docs/*.md` architecture roadmaps, public headers in `include/`,
  and extracted function/builtin signatures — into a local vector store
  (e.g. FAISS/Chroma). Each chat request retrieves the top-k relevant
  chunks and injects only those into the `system_prompt`.
  *Verbatim injection of every `.cpp`/`.h` (the runtime alone is
  ~28 kLOC) is infeasible per-request — see §7, FIX-7.*
* **Agent scope.** Answer architecture questions, guide users through
  LLVM/MLIR errors surfaced by the compiler, and autocomplete valid
  mathematical MATLAB blocks for the supported subset.

### F5. Pocket-MATLAB I/O: Workspace, Files, Plots & Artifacts

This is the layer that makes the product feel like "MATLAB in your
pocket" for mobile clients. None of it requires compiler changes — it is
all FastAPI orchestration over the existing `matlabc` runtime.

* **Per-user workspaces.** Each user/session owns a subtree under the
  persistent volume, e.g. `/workspace/{user_id}/{session_id}/`. The REPL
  and codegen child processes run with their cwd confined to it, so
  relative paths in MATLAB code resolve there and data never leaks
  across users.
* **File upload (data in).** `POST /v1/files` (multipart) stores CSV /
  data files into the caller's workspace; user code then reads them with
  `readmatrix('data.csv')` / `readtable('data.csv')` (auto-delimiter +
  type inference) or `fopen`/`load`. Size and type limits enforced.
* **File listing & download (results out).** `GET /v1/files` lists the
  workspace; `GET /v1/files/{path}` downloads any artifact — uploaded
  data, generated code (saved from a codegen call), `save`'d `.mat`
  results, or `writematrix`/`writetable` outputs.
* **Plotting.** Build with `-DMATLAB_LLVM_WITH_PLOT=ON`. Two delivery
  modes: (a) user code calls `saveas(gcf,'fig.png')` → served as a
  workspace artifact; (b) a `POST /v1/plot` convenience route runs a
  snippet and returns the rendered figure bytes (PNG/SVG/PDF) directly,
  using the in-memory `matlab_render_png/svg/pdf` C-ABI. Mobile clients
  display the image inline and offer "download / share".
* **REPL figure capture.** When a REPL turn produces a figure, the
  `/v1/repl` response includes references to the generated image
  artifacts alongside the text output, so the app can render plots
  produced interactively.

---

## 4. Non-Functional Requirements

* **Performance.** `< 200 ms` applies to **`/v1/check`** (parse + Sema)
  only. JIT execution (`/v1/repl`) and codegen pay process-start +
  MLIR/LLVM JIT/lowering cost; budget **p95 < ~1.5 s** for typical
  programs, reduced via a warm REPL-worker pool (FIX-8).
* **Process isolation & untrusted-code execution.** The REPL/DAP paths
  JIT-compile and run **arbitrary user code**. Each invocation runs as a
  child process under `subprocess` with: a hard **wall-clock timeout**,
  CPU/memory/file-size **`rlimit`s**, a non-root user, a working
  directory confined to the session's `/workspace` subtree, and (where
  available) a syscall sandbox (`nsjail`/`firejail`/gVisor) or an
  ephemeral per-session container. Crashes (core dumps), infinite loops,
  and OOM are contained to the child and never take down FastAPI
  (FIX-9).
* **Security & network.** Coolify's Traefik enforces TLS (HTTPS/WSS)
  end-to-end. The chat and codegen APIs require an auth token; CORS is
  restricted to the iPad client origins.

---

## 5. Baseline Environment Configuration (Code Blueprint)

The blueprints below are the corrected versions. The original draft's
`docker-compose.yaml` declared an obsolete `version:` key and exposed a
nonexistent DAP port; the Dockerfile cloned no real URL, targeted
LLVM 18, and dropped the runtime shared libraries.

### `docker-compose.yaml` (Coolify-ready)

```yaml
# Compose v2: no top-level `version:` key.
services:
  backend-compiler:
    build:
      context: .
      dockerfile: Dockerfile
    image: matlab-llvm-backend:latest
    container_name: matlab_llvm_api
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=8000
      # Resource ceilings for the untrusted-code child processes.
      - REPL_TIMEOUT_SECONDS=10
      - REPL_MEMORY_MB=512
    ports:
      - "8000:8000"   # REST + MCP/SSE + DAP-over-WebSocket (single port)
    volumes:
      - workspace_data:/workspace
    restart: always
    # Defense-in-depth for arbitrary user code.
    security_opt:
      - no-new-privileges:true
    # cap_drop / read_only / pids_limit recommended (see plan §6).

volumes:
  workspace_data:
```

> Removed: `version: '3.8'` (obsolete), the `4711:4711` mapping (no DAP
> listener exists — DAP rides the WebSocket on `:8000`).

### `Dockerfile` (multi-stage)

```dockerfile
# --- STAGE 1: BUILD THE C++ PROJECT ---------------------------------------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Pin the LLVM/MLIR version to match what the source compiles against
# (dev env uses 22; 20 is a safe stable floor). Ubuntu's default repos
# do not carry MLIR dev packages for 20+, so use apt.llvm.org.
ARG LLVM_VERSION=20
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget lsb-release software-properties-common \
        cmake ninja-build git build-essential pkg-config \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] \
        http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev libllvm${LLVM_VERSION} \
        libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools \
        libzstd-dev zlib1g-dev \
        libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Coolify provides the repo as the build context — COPY it, do not clone.
COPY . .
# Plotting ON so mobile clients get PNG/SVG/PDF figures (Cairo backend).
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMATLAB_LLVM_WITH_PLOT=ON \
        -DLLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm \
        -DMLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir \
    && cmake --build build -j

# --- STAGE 2: LEAN RUNTIME ------------------------------------------------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=20

# matlabc dynamically links libLLVM/libMLIR — install the *runtime* libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget python3 python3-venv python3-pip \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] \
        http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libllvm${LLVM_VERSION} libmlir-${LLVM_VERSION} libzstd1 zlib1g \
        libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/matlabc /usr/local/bin/matlabc
# Ship the runtime sources/docs that the RAG indexer and codegen need.
COPY --from=builder /app/docs   /app/source_context/docs
COPY --from=builder /app/include /app/source_context/include
COPY --from=builder /app/runtime /app/source_context/runtime

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ /app/server/

# Run as non-root for the untrusted-code paths.
RUN useradd -m runner && mkdir -p /workspace && chown runner /workspace
USER runner

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> `libmlir-<v>` (runtime) may be named `libmlir-<v>` or `libmlirc<v>`
> depending on the apt.llvm.org snapshot; verify with
> `apt-cache search libmlir` and pin accordingly. If the package is
> unavailable for the chosen version, copy the `.so` files from the
> builder's `/usr/lib/llvm-<v>/lib/` instead, or statically link
> `matlabc` (`-DLLVM_LINK_LLVM_DYLIB=OFF` against static LLVM/MLIR).

---

## 6. Validation & Acceptance Plan

1. **Automated route tests (iPad).** Fire requests from an iOS HTTP
   client validating JSON payloads across `check`, `repl`, and each
   `codegen/*` target.
2. **MCP tool tests.** Connect an MCP-compatible client and confirm
   natural-language invocation correctly triggers the compiler tools
   over `/mcp/sse`.
3. **Debugger stress test.** Set breakpoints in non-trivial MATLAB
   scripts from VS Code Web on the iPad via the `/v1/dap/ws/{sid}`
   bridge; verify stops, variable inspection, stepping, and reverse
   stepping against the in-container session.
4. **Isolation test (new).** Submit a deliberately crashing program, an
   infinite loop, and an OOM allocator; confirm each is killed by the
   timeout/`rlimit`, returns a clean error, and **never** degrades the
   FastAPI process or other sessions.
5. **Cold/warm latency test (new).** Measure `/v1/check` p95 (`<200ms`)
   and `/v1/repl` p95 (warm-pool target) under concurrent load.

---

## 7. Feasibility Assessment & Corrections

**Verdict: feasible.** Every capability the product needs already
exists in `matlabc` (REPL, codegen, native DAP). The original draft is
sound in shape but contains several technical errors that would break
the build or the runtime. Each correction below is backed by direct
inspection of the repository.

| # | Original draft | Problem | Correction | Evidence |
|---|---|---|---|---|
| **FIX-1** | "Container 2: `lldb-dap` service" on "Port 4711 (DAP WebSocket)"; WS proxy "optional". | `matlabc` *is* the DAP adapter; there is no `lldb-dap` and **no DAP network port** — transport is stdio. | Drop `lldb-dap`. Run `matlabc -dap <prog>` as a per-session child; bridge its stdio to the **mandatory** `/v1/dap/ws/{sid}` WebSocket. | `matlabc -dap` mode in `tools/matlabc/main.cpp`; `docs/debug.md` ("reads JSON-RPC frames from stdin"); `grep AF_INET\|socket(\|bind(\|listen(` over `lib/`+`tools/` → **no matches**; `test/Debug/dap_client.py` spawns `matlabc -dap` over `stdin/stdout`. |
| **FIX-2** | Dockerfile installs `clang-18 / llvm-18-dev`. | Project targets LLVM 20–22; 18 will fail to configure/compile. | Pin one matching version (20 stable floor; 22 = current dev). Install via `apt.llvm.org`. | `find_package(LLVM/MLIR CONFIG)` in `CMakeLists.txt`; dev env `otool -L build/matlabc` → `libLLVM ... current version 22.1.3`; README badge says MLIR LLVM 20 (stale). |
| **FIX-3** | Prose: stage 1 "compiles LLVM 22.x, MLIR…". | Building LLVM+MLIR from source is multi-hour / 16 GB+ RAM — impractical in CI/Coolify. | Install prebuilt LLVM+MLIR dev packages; compile **only** `matlab_llvm`. | `find_package(... CONFIG REQUIRED)` consumes a prebuilt install; no LLVM sources are vendored in-tree. |
| **FIX-4** | Runtime stage copies only `/app/build/matlabc`. | `matlabc` **dynamically links** `libLLVM`/`libMLIR`; binary alone won't start. | Install matching `libllvm<v>`/`libmlir-<v>` runtime libs in stage 2 (or copy `.so`s / static-link). | `otool -L build/matlabc` → `@... libMLIR.dylib`, `libLLVM.dylib`. |
| **FIX-5** | "`matlabc -repl` returning clean buffers in JSON format". | REPL emits **plain text**; default no-flag mode is `-check` (no execution); no "run file" mode. | FastAPI wraps captured stdout/stderr in JSON. Drive `-repl` over stdin. Use `-check` for validation. | `docs/repl.md`; `Mode Mode = Mode::Check;` default in `tools/matlabc/main.cpp`. |
| **FIX-6** | One route `/codegen/python` described as "TypeScript/Python". | Python and TypeScript are distinct flags. | Separate routes per target: `python`, `typescript`, `cpp`, `c`, `systemverilog`. | `-emit-python` vs `-emit-typescript`/`-emit-ts` in `main.cpp`. |
| **FIX-7** | RAG injects whole critical `.cpp`/`.h` into every `system_prompt`. | Runtime alone ~28 kLOC; per-request whole-file stuffing busts context + cost. | Build a startup vector index (chunk+embed) over `docs/` + `include/` + signatures; retrieve top-k per query. | README ("~28 kLOC C++ runtime"); rich `docs/*.md` roadmaps already exist as ideal RAG corpus. |
| **FIX-8** | `<200ms` for all REST validations. | JIT/codegen exceed 200 ms (process start + MLIR lowering). | Scope `<200ms` to `/v1/check`; warm-pool REPL workers; p95 < ~1.5 s for repl/codegen. | REPL = in-process JIT compile per turn (`docs/repl.md`). |
| **FIX-9** | Isolation = `subprocess` capture only. | REPL/DAP run **arbitrary** native code; need resource + syscall limits. | Add timeouts, `rlimit`s, non-root, confined cwd, optional `nsjail`/gVisor or per-session container. | DAP "JIT-executes in-process"; runtime exposes file IO + image codecs etc. |
| **FIX-10** | `version: '3.8'`; `4711:4711`; `git clone https://github.com .`; code-server in diagram only. | Obsolete key; phantom port; malformed clone; missing service. | Compose v2 (no `version:`); single `:8000`; `COPY . .`; code-server marked optional sidecar. | Compose spec; FIX-1; Dockerfile line in original draft. |

### Things the draft got right (kept as-is)

* `matlabc -repl` JIT REPL — exists and is stateful across inputs.
* Codegen to Python / TypeScript / C / C++ / SystemVerilog — all exist.
* Step-debugging via DAP — exists (and is richer than assumed:
  conditional breakpoints, logpoints, reverse stepping, matrix memory
  inspection).
* Coolify + Docker Compose + Traefik TLS + persistent `/workspace`.
* OpenAI-compatible `/v1/chat/completions` with internal code context —
  feasible, with the retrieval correction in FIX-7.

### Open product decisions (carried into the plan)

* **Stateful vs stateless REPL sessions** — one-shot per request, or
  long-lived per-user workspace? (affects warm-pool + session store).
* **Auth model** — static API key, per-user tokens, or Coolify-fronted
  basic auth?
* **Chat backend** — proxy to OpenAI (needs key + egress) vs a local
  model. The current draft assumes OpenAI.
* **code-server** — ship the optional IDE sidecar in v1 or defer?
