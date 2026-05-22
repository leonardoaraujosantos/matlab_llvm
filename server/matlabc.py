"""Thin async wrapper around the real ``matlabc`` CLI.

Verified invocations (see tools/matlabc/main.cpp):
  * check   -> ``matlabc <file.m>``   (Check is the DEFAULT mode; there is
                no ``-check`` flag — passing one errors as "unknown flag").
  * repl    -> ``matlabc -repl``      (program text read from STDIN to EOF).
  * codegen -> ``matlabc -emit-<t> <file.m>`` (emitted source on STDOUT).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from config import settings
from sandbox import SandboxResult
from sandbox import run as sandbox_run

# REST target -> matlabc emit flag.
EMIT_FLAGS: dict[str, str] = {
    "python": "-emit-python",
    "typescript": "-emit-typescript",
    "c": "-emit-c",
    "cpp": "-emit-cpp",
    "systemverilog": "-emit-systemverilog",
}


def _bin() -> str:
    return str(settings.matlabc_path)


def _write_snippet(src: str, workspace: Path) -> Path:
    path = workspace / f"snippet_{uuid.uuid4().hex[:12]}.m"
    path.write_text(src)
    return path


async def check(src: str, workspace: Path) -> SandboxResult:
    """Validate-only (default mode, no execution)."""
    f = _write_snippet(src, workspace)
    return await sandbox_run([_bin(), str(f)], cwd=workspace)


async def repl(src: str, workspace: Path) -> SandboxResult:
    """JIT-execute statements; program text is piped over stdin."""
    return await sandbox_run([_bin(), "-repl"], cwd=workspace, stdin=src)


async def emit(target: str, src: str, workspace: Path) -> SandboxResult:
    """Transpile to one of the EMIT_FLAGS targets."""
    flag = EMIT_FLAGS[target]
    f = _write_snippet(src, workspace)
    return await sandbox_run([_bin(), flag, str(f)], cwd=workspace)
