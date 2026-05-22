"""Hardened async subprocess launcher for untrusted matlabc children.

Every REPL/codegen/check/DAP invocation runs arbitrary native code, so
each child is confined: rlimits (CPU, memory, file size, process count),
a new session/process-group we can kill wholesale, a scrubbed env, a cwd
jail (the session workspace), a hard wall-clock timeout, and a byte cap
on captured output.

`preexec_fn` / `os.setsid` are POSIX-only; this targets Linux (container)
and macOS (local dev). RLIMIT_AS is unreliable on macOS, so the container
is the real memory boundary — see plan §7.
"""

from __future__ import annotations

import asyncio
import os
import resource
import signal
from dataclasses import dataclass
from pathlib import Path

import limits
from config import settings

# Env vars that let matlabc find its dynamically-linked libs (libLLVM /
# libMLIR / libcairo). Preserved through the scrub; everything else dropped.
_PRESERVE_ENV = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")


@dataclass
class SandboxResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _make_preexec():
    cpu = settings.cpu_seconds
    mem = settings.memory_mb * 1024 * 1024
    fsize = settings.file_size_mb * 1024 * 1024
    nproc = settings.max_procs

    def _apply() -> None:
        # New session => child is process-group leader (pgid == pid), so a
        # single killpg() reaps the whole tree on timeout.
        os.setsid()
        # Soft CPU limit fires SIGXCPU; +1 hard gives the handler no slack.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        if mem > 0:  # 0 disables; AS over-counts libLLVM mmaps (see config)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass  # not enforced on some platforms (macOS)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError, AttributeError):
            pass

    return _apply


def _child_env(workspace: Path) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workspace),
        "TMPDIR": str(workspace),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for k in _PRESERVE_ENV:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _cap(buf: bytes | None, cap: int) -> tuple[str, bool]:
    buf = buf or b""
    truncated = len(buf) > cap
    return buf[:cap].decode("utf-8", "replace"), truncated


async def run(
    argv: list[str],
    *,
    cwd: str | os.PathLike,
    stdin: str | bytes | None = None,
    timeout: float | None = None,
    output_cap: int | None = None,
) -> SandboxResult:
    """Launch ``argv`` confined to the sandbox and return the result."""
    timeout = settings.wall_timeout_s if timeout is None else timeout
    output_cap = settings.output_cap_bytes if output_cap is None else output_cap
    stdin_bytes = stdin.encode() if isinstance(stdin, str) else stdin

    timed_out = False
    async with limits.job_semaphore():  # global concurrency cap
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=_child_env(Path(cwd)),
            preexec_fn=_make_preexec(),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            _kill_group(proc.pid)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                out, err = b"", b""

    stdout, so_trunc = _cap(out, output_cap)
    stderr, se_trunc = _cap(err, output_cap)
    return SandboxResult(
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        stdout_truncated=so_trunc,
        stderr_truncated=se_trunc,
    )


async def spawn(
    argv: list[str],
    *,
    cwd: str | os.PathLike,
    stdin: int = asyncio.subprocess.PIPE,
    stdout: int = asyncio.subprocess.PIPE,
    stderr: int = asyncio.subprocess.PIPE,
) -> asyncio.subprocess.Process:
    """Spawn a confined *long-lived* child (rlimits + new session + scrubbed
    env + cwd jail) and hand back the Process for the caller to drive.

    Use this for interactive sessions (DAP) where ``run``'s
    communicate/timeout model doesn't fit. Call :func:`terminate` to kill the
    whole process group when the session ends.
    """
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=str(cwd),
        env=_child_env(Path(cwd)),
        preexec_fn=_make_preexec(),
    )


def terminate(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group, if still running."""
    if proc.returncode is None:
        _kill_group(proc.pid)
