"""Env-driven settings for the remote backend.

All settings are overridable via environment variables prefixed with
``MATLAB_BACKEND_`` (e.g. ``MATLAB_BACKEND_MATLABC_BIN``) or a ``.env``
file in the working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the server/ directory this file lives in. Used to
# locate the just-built matlabc binary independent of the process cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MATLAB_BACKEND_",
        env_file=".env",
        extra="ignore",
    )

    # --- Compiler binary ---------------------------------------------------
    # Path to the matlabc CLI. Defaults to the repo's `just build` output;
    # the justfile recipes set this explicitly to an absolute path.
    matlabc_bin: str = str(_REPO_ROOT / "build" / "matlabc")

    # --- Workspaces --------------------------------------------------------
    # Root under which per-user/session workspaces are created. Each child
    # process runs with its workspace as cwd.
    workspace_root: str = "/tmp/matlab_llvm_workspaces"

    # --- Resource ceilings (applied to every child process) ----------------
    # NOTE: RLIMIT_AS counts *virtual* address space, which over-counts the
    # large libLLVM/libMLIR mappings of a JIT process — set generously (or 0
    # to disable) and treat the container memory cgroup as the real boundary.
    cpu_seconds: int = 10            # RLIMIT_CPU (seconds of CPU time)
    memory_mb: int = 2048            # RLIMIT_AS (address space; 0 = disabled)
    file_size_mb: int = 64           # RLIMIT_FSIZE (max file written)
    max_procs: int = 64              # RLIMIT_NPROC (fork bomb guard)
    wall_timeout_s: float = 20.0     # hard wall-clock kill
    output_cap_bytes: int = 1_000_000  # max captured stdout/stderr bytes

    # --- Uploads (Phase 3) -------------------------------------------------
    max_upload_mb: int = 25

    # --- Hardening (Phase 7) -----------------------------------------------
    max_concurrent_jobs: int = 8        # global cap on concurrent matlabc children
    rate_limit_per_minute: int = 120    # per-client request cap (0 disables)
    user_quota_mb: int = 200            # per-workspace disk quota (0 disables)

    # --- Stateful REPL (Phase 7 / plan §11) --------------------------------
    repl_stateful: bool = True          # keep a long-lived matlabc -repl per session
    repl_idle_timeout_s: float = 900.0  # evict sessions idle longer than this
    repl_evict_interval_s: float = 60.0  # background sweep cadence
    warm_pool_size: int = 2             # pre-warmed matlabc -repl workers (0 disables)

    # --- Tier-2 syscall sandbox (plan Phase 7) -----------------------------
    # Wrap children in an OS sandbox for syscall-level isolation beyond
    # rlimits. Linux-only; falls back to rlimit-only if the tool is missing.
    sandbox_backend: str = "none"       # none | bwrap | firejail | nsjail
    sandbox_allow_net: bool = False     # untrusted code gets no network by default

    # --- Plot (Phase 3 / TRD F5) -------------------------------------------
    plot_default_format: str = "png"    # png | svg | pdf

    # --- Server bind -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Auth -------------------------------------------------------------
    # Three modes, resolved by `auth_mode`:
    #   cyberdyne  — validate bearer tokens against CyberdyneAuth (preferred)
    #   token      — a single shared static bearer token
    #   none       — open (local-dev default)
    api_token: str = ""
    cyberdyne_auth_url: str = Field(
        "", validation_alias=AliasChoices("CYBERDYNE_AUTH_URL", "MATLAB_BACKEND_CYBERDYNE_AUTH_URL")
    )
    auth_verify_cache_ttl_s: int = 30   # cache successful /users/me checks
    cyberdyne_timeout_s: float = 5.0

    # --- MCP auth ----------------------------------------------------------
    # When on, /mcp requires a backend-minted bearer token (POST /v1/mcp/token,
    # itself authenticated). Tokens are stateless HMAC-signed and bound to the
    # caller's identity.
    mcp_require_auth: bool = False
    mcp_token_secret: str = ""           # HMAC key; falls back to api_token / dev key
    mcp_token_ttl_s: int = 2592000       # 30 days

    # --- RAG / chat (Phase 6) ----------------------------------------------
    # Root holding the docs corpus to index (`<root>/docs/**/*.md`). In the
    # container this is /app/source_context; locally it's the repo root.
    source_context_root: str = str(_REPO_ROOT)
    rag_enabled: bool = True
    rag_top_k: int = 4
    rag_max_chunk_chars: int = 4000
    # OpenAI proxy. Read the bare names too (compose passes OPENAI_API_KEY).
    openai_api_key: str = Field("", validation_alias=AliasChoices("OPENAI_API_KEY", "MATLAB_BACKEND_OPENAI_API_KEY"))
    openai_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "MATLAB_BACKEND_OPENAI_BASE_URL"),
    )
    openai_model: str = Field(
        "gpt-4o-mini", validation_alias=AliasChoices("OPENAI_MODEL", "MATLAB_BACKEND_OPENAI_MODEL")
    )

    @property
    def auth_mode(self) -> str:
        if self.cyberdyne_auth_url:
            return "cyberdyne"
        if self.api_token:
            return "token"
        return "none"

    @property
    def matlabc_path(self) -> Path:
        return Path(self.matlabc_bin).expanduser()

    @property
    def workspace_root_path(self) -> Path:
        return Path(self.workspace_root).expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
