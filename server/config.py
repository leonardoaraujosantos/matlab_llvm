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

    # --- Server bind -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Auth (optional; empty disables, the local-dev default) ------------
    api_token: str = ""

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
    def matlabc_path(self) -> Path:
        return Path(self.matlabc_bin).expanduser()

    @property
    def workspace_root_path(self) -> Path:
        return Path(self.workspace_root).expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
