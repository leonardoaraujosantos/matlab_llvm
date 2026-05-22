"""Pydantic request/response schemas shared across routers."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class ReplResponse(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
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
