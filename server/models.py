"""Pydantic request/response schemas shared across routers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionMixin(BaseModel):
    user_id: str | None = Field(default=None, description="Caller identity; drives workspace routing.")
    session_id: str | None = Field(default=None, description="Session key; isolates workspace state.")


class Diagnostic(BaseModel):
    severity: str
    message: str
    file: str | None = None
    line: int | None = None
    col: int | None = None


# --- /v1/check -----------------------------------------------------------
class CheckRequest(SessionMixin):
    source: str = Field(..., description="MATLAB source to validate.")


class CheckResponse(BaseModel):
    ok: bool
    diagnostics: list[Diagnostic] = []
    stdout: str = ""
    stderr: str = ""


# --- /v1/repl ------------------------------------------------------------
class ReplRequest(SessionMixin):
    source: str = Field(..., description="MATLAB source/statements to execute.")
    stateful: bool | None = Field(
        default=None, description="Override the server's stateful-REPL default for this call."
    )


class ReplResponse(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    stateful: bool = False
    artifacts: list[str] = Field(default=[], description="New figure/file paths (download via /v1/files).")


# --- /v1/codegen/{target} ------------------------------------------------
class CodegenRequest(SessionMixin):
    source: str = Field(..., description="MATLAB source to transpile.")


class CodegenResponse(BaseModel):
    ok: bool
    language: str
    code: str = ""
    diagnostics: list[Diagnostic] = []
    stderr: str = ""


# --- /v1/files -----------------------------------------------------------
class FileInfo(BaseModel):
    path: str
    size: int
    modified: float


class FileListResponse(BaseModel):
    files: list[FileInfo] = []


class UploadResponse(BaseModel):
    ok: bool
    file: FileInfo


# --- /v1/auth/whoami -----------------------------------------------------
class WhoAmIResponse(BaseModel):
    authenticated: bool
    mode: str
    id: str | None = None
    email: str | None = None
    organization_id: str | None = None


# --- /v1/mcp/token -------------------------------------------------------
class McpTokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    subject: str
    expires_at: int


# --- /v1/plot ------------------------------------------------------------
class PlotRequest(SessionMixin):
    # No pattern/regex constraint: any source accepted by /v1/repl is accepted
    # here verbatim (the server only appends a saveas). See issue #56.
    source: str = Field(..., description="MATLAB plotting snippet (server appends saveas).")
    format: str | None = Field(default=None, description="png | svg | pdf")


class PlotResponse(BaseModel):
    """ReplResponse-shaped result for /v1/plot.

    The produced figure path is surfaced in ``artifacts`` (download the bytes
    via ``GET /v1/files/{path}``), mirroring /v1/repl. Pass ``?raw=true`` (or an
    image/pdf ``Accept`` header) to stream the figure bytes back directly.
    """

    ok: bool
    format: str = Field(description="png | svg | pdf")
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    artifacts: list[str] = Field(
        default=[], description="Produced figure path(s); download via /v1/files/{path}."
    )


# --- /v1/chat/completions (OpenAI-compatible) ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    # extra="allow" lets OpenAI params (top_p, max_tokens, ...) pass through
    # to the upstream proxy untouched.
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
