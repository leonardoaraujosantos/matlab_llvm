# Compiler Server Spec

## Purpose
The compiler server is a FastAPI web service that exposes the `matlabc` CLI over HTTP, WebSocket, and the MCP protocol. It lets clients validate, JIT-execute, plot, transpile (codegen), and debug MATLAB-subset source remotely, with each invocation confined to a per-user/per-session workspace and run inside a hardened subprocess sandbox. It also serves an OpenAI-compatible chat endpoint grounded in the project docs via a built-in retrieval (RAG) index. (src: server/main.py, src: server/README.md)

## Requirements

### Requirement: HTTP and WebSocket API surface
The system SHALL expose the compiler over a versioned `/v1` REST API plus a DAP WebSocket bridge, an MCP mount, and a health endpoint, registering the routers `check`, `repl`, `codegen`, `files`, `chat`, `plot`, `whoami`, `mcp_token`, and `dap_ws`. (src: server/main.py, src: server/routers/)

#### Scenario: Core REST endpoints are mounted
- **WHEN** the FastAPI app is created
- **THEN** the system SHALL serve `POST /v1/check`, `POST /v1/repl`, `POST /v1/codegen/{target}`, `POST /v1/plot`, `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{path}`, `POST /v1/chat/completions`, `GET /v1/auth/whoami`, `POST /v1/mcp/token`, the WebSocket `/v1/dap/ws/{session_id}`, the MCP sub-app mounted at `/mcp`, and `GET /healthz`

#### Scenario: Health endpoint reports runtime state
- **WHEN** a client calls `GET /healthz`
- **THEN** the system SHALL return `status: ok` together with the configured `matlabc` path, whether the binary is present, and the runtime sandbox backend/active/allow_net (including a `reason` when the tier-2 sandbox probe failed)

### Requirement: Authentication and authorization
The system SHALL gate all `/v1` routes behind `require_auth` in one of three modes selected from configuration: `cyberdyne` (bearer validated against CyberdyneAuth `GET /api/v1/users/me`), `token` (a single shared static bearer), or `none` (open). (src: server/auth.py, src: server/config.py, src: server/main.py)

#### Scenario: Missing or invalid bearer in token mode
- **WHEN** `auth_mode` is `token` and a request to a `/v1` route carries no bearer or a bearer that does not equal the configured `api_token`
- **THEN** the system SHALL reject the request with HTTP 401 and a `WWW-Authenticate: Bearer` header

#### Scenario: Verified Cyberdyne identity becomes the principal
- **WHEN** `auth_mode` is `cyberdyne` and a valid, active bearer is presented
- **THEN** the system SHALL set the request principal to the verified identity id (caching the verification briefly), so workspaces resolve per identity regardless of any client-supplied `user_id`

#### Scenario: Auth service unavailable
- **WHEN** `auth_mode` is `cyberdyne` and the CyberdyneAuth call fails to connect or returns 5xx
- **THEN** the system SHALL respond with HTTP 503

### Requirement: Per-identity workspace isolation and routing
The system SHALL resolve every operation to a sanitised workspace directory under `WORKSPACE_ROOT` keyed by `(user_id, session_id)`, where the authenticated principal overrides any client-supplied `user_id`, and SHALL reject path traversal outside that workspace. (src: server/workspaces.py, src: server/principal.py, src: server/services.py)

#### Scenario: Identifier sanitisation prevents escape
- **WHEN** a workspace is resolved for a given user/session pair
- **THEN** the system SHALL replace unsafe characters, strip leading dots, cap each component length, and create the directory under the workspace root

#### Scenario: Path traversal rejected
- **WHEN** a file upload, download, or read references a path that resolves outside the session workspace
- **THEN** the system SHALL raise an error rather than access the file (`resolve_in_workspace` raises `ValueError`, surfaced as HTTP 400)

### Requirement: Sandboxed subprocess execution with resource limits
The system SHALL run every `matlabc` child in a confined subprocess: a new session/process-group, scrubbed environment with the workspace as cwd, rlimits (CPU, address space, file size, process count), a global concurrency cap, a wall-clock timeout that kills the whole process group, and a byte cap on captured output; with an optional tier-2 OS sandbox (`bwrap`/`firejail`/`nsjail`) layered on when configured. (src: server/sandbox.py, src: server/jail.py, src: server/limits.py, src: server/config.py)

#### Scenario: Timeout kills the process group
- **WHEN** a child exceeds the configured wall-clock timeout
- **THEN** the system SHALL SIGKILL the child's process group and return a result marked `timed_out`

#### Scenario: Tier-2 sandbox self-test and fallback
- **WHEN** a tier-2 sandbox backend is configured but its startup probe fails (e.g. the host blocks unprivileged user namespaces)
- **THEN** the system SHALL downgrade the runtime backend to `none` (rlimit-only) and surface the failure reason on `/healthz`

#### Scenario: Per-client rate limiting
- **WHEN** a client exceeds `rate_limit_per_minute` requests within the sliding 60-second window
- **THEN** the system SHALL respond with HTTP 429 and a `Retry-After` header

### Requirement: Compile, run, plot, and codegen job execution
The system SHALL drive the `matlabc` CLI for check (default mode, no execution), repl (`-repl`, source over stdin), plot (repl run with an appended `saveas`), and codegen (`-emit-<target>` for one of python, typescript, c, cpp, systemverilog), returning structured diagnostics parsed from compiler output. (src: server/matlabc.py, src: server/services.py, src: server/routers/, src: server/diagnostics.py)

#### Scenario: Codegen rejects an unknown target
- **WHEN** `POST /v1/codegen/{target}` is called with a target not in the emit-flag map
- **THEN** the system SHALL respond with HTTP 404 naming the supported targets

#### Scenario: Plot returns the produced figure
- **WHEN** `POST /v1/plot` runs a snippet that produces a non-empty figure file
- **THEN** the system SHALL return the figure path in `artifacts` (downloadable via `GET /v1/files/{path}`) as JSON, or stream the raw bytes when `?raw=true` or an image/pdf `Accept` header is sent; and SHALL respond with HTTP 422 when no figure was produced

#### Scenario: Diagnostics are parsed from compiler output
- **WHEN** `matlabc` emits clang-style `file:line:col: severity: message` (or bare `severity: message`) lines
- **THEN** the system SHALL return them as structured `Diagnostic` entries alongside the raw stdout/stderr

### Requirement: Session and workspace lifecycle management
The system SHALL support stateful long-lived `matlabc -repl` sessions per `(user, session)` backed by a pre-warmed worker pool, so workspace variables persist across `/v1/repl` calls, and SHALL evict idle sessions on a background sweep. (src: server/sessions.py, src: server/config.py, src: server/main.py)

#### Scenario: Stateful turn persists state and surfaces artifacts
- **WHEN** stateful REPL is enabled and a turn runs against a live session
- **THEN** the system SHALL execute it in the session's adopted workspace, return new artifacts created during the turn, and keep the worker alive for the next turn

#### Scenario: Idle and broken sessions are recycled
- **WHEN** a session is idle longer than the idle timeout, or a turn times out / the child has exited
- **THEN** the system SHALL terminate and retire the session, migrating its workspace files back to the deterministic directory so they remain reachable via `/v1/files`

### Requirement: File upload, listing, and download
The system SHALL accept multipart uploads into the session workspace, list the workspace tree, and stream files back, enforcing a per-upload size cap and an optional per-workspace disk quota. (src: server/routers/files.py, src: server/limits.py, src: server/config.py)

#### Scenario: Upload exceeds the size cap or quota
- **WHEN** an upload exceeds `max_upload_mb` or would push the workspace past `user_quota_mb`
- **THEN** the system SHALL delete the partial file and respond with HTTP 413

#### Scenario: Download of a missing file
- **WHEN** `GET /v1/files/{path}` targets a path that is not a regular file in the workspace
- **THEN** the system SHALL respond with HTTP 404

### Requirement: MCP tools interface
The system SHALL expose the compiler to MCP/AI clients over streamable-HTTP at `/mcp` via FastMCP, offering the tools `matlab_check`, `matlab_repl`, `matlab_codegen`, `list_files`, and `read_file`, sharing the same sandbox and workspace-isolation layer as the REST API, and SHALL optionally require a backend-minted HMAC bearer token. (src: server/mcp_tools.py, src: server/mcp_auth.py, src: server/routers/mcp_token.py)

#### Scenario: Minted token binds MCP calls to an identity
- **WHEN** an authenticated caller mints a token via `POST /v1/mcp/token` and presents it on `/mcp`
- **THEN** the system SHALL verify the token locally (HMAC, type `mcp`, unexpired) and bind the MCP request principal to the token subject so tools run against that user's workspace

#### Scenario: MCP auth required but token invalid
- **WHEN** `mcp_require_auth` is enabled and a request to `/mcp` presents a tampered or expired token
- **THEN** the system SHALL reject the request (the token verifier returns no access token)

### Requirement: Documentation-grounded chat (RAG)
The system SHALL build a dependency-free BM25 index over `<source_context_root>/docs/**/*.md` at startup and serve an OpenAI-compatible `POST /v1/chat/completions` that retrieves top-k doc chunks for the user's last message and prepends a cited context block, proxying upstream when an OpenAI key is configured and returning a retrieval-only answer otherwise. (src: server/rag.py, src: server/routers/chat.py, src: server/config.py)

#### Scenario: Offline retrieval-only answer
- **WHEN** `POST /v1/chat/completions` is called and no OpenAI API key is configured
- **THEN** the system SHALL return a retrieval-only completion built from the most relevant indexed documentation, including the source citations

#### Scenario: Upstream proxy with citations
- **WHEN** an OpenAI key is configured
- **THEN** the system SHALL prepend the cited context as a system message and proxy the request upstream, supporting streaming passthrough and exposing the citation list to the client
