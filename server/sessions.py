"""Stateful REPL sessions (plan §11 / Phase 7).

Keeps a long-lived ``matlabc -repl`` child per (user, session) so workspace
variables persist across ``/v1/repl`` calls. ``matlabc -repl`` prints no
prompt over a pipe, so each turn is delimited by appending a unique
``disp('<marker>')`` and reading stdout until the marker appears. stderr is
merged into stdout (``stderr=STDOUT``) so a turn's errors are captured inline.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import limits
import sandbox
from config import settings


@dataclass
class ReplSession:
    proc: asyncio.subprocess.Process
    workspace: Path
    lock: asyncio.Lock
    last_used: float


async def _read_until(stream: asyncio.StreamReader, marker: bytes, timeout: float) -> tuple[str, bool]:
    buf = bytearray()

    async def loop() -> None:
        while marker not in buf:
            chunk = await stream.read(4096)
            if not chunk:  # EOF before marker
                return
            buf.extend(chunk)

    timed_out = False
    try:
        await asyncio.wait_for(loop(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
    text = bytes(buf).decode("utf-8", "replace")
    idx = text.find(marker.decode("utf-8", "replace"))
    if idx != -1:
        text = text[:idx]
    return text, timed_out


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ReplSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(user_id: str | None, session_id: str | None) -> str:
        return f"{user_id or 'anon'}::{session_id or 'default'}"

    @property
    def active(self) -> int:
        return len(self._sessions)

    async def _spawn(self, workspace: Path) -> ReplSession:
        proc = await sandbox.spawn(
            [str(settings.matlabc_path), "-repl"],
            cwd=workspace,
            stderr=asyncio.subprocess.STDOUT,
        )
        return ReplSession(proc=proc, workspace=workspace, lock=asyncio.Lock(), last_used=time.time())

    async def run_turn(
        self, user_id: str | None, session_id: str | None, workspace: Path, source: str, timeout: float
    ) -> dict:
        key = self._key(user_id, session_id)
        async with self._lock:
            sess = self._sessions.get(key)
            if sess is None or sess.proc.returncode is not None:
                sess = await self._spawn(workspace)
                self._sessions[key] = sess

        async with sess.lock:
            marker = f"<<<MLBC_TURN_{uuid.uuid4().hex}>>>"
            payload = source.rstrip("\n") + "\n" + f"disp('{marker}')\n"
            async with limits.job_semaphore():
                sess.proc.stdin.write(payload.encode("utf-8"))
                await sess.proc.stdin.drain()
                out, timed_out = await _read_until(sess.proc.stdout, marker.encode("utf-8"), timeout)
            sess.last_used = time.time()

        alive = sess.proc.returncode is None
        # A timed-out / dead session has an unknown stream state — recycle it.
        if timed_out or not alive:
            await self._drop(key)
            alive = False
        return {"stdout": out, "timed_out": timed_out, "alive": alive}

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        # Wait within the running loop so the subprocess transport closes
        # cleanly (avoids "Event loop is closed" on interpreter teardown).
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    async def _drop(self, key: str) -> None:
        async with self._lock:
            sess = self._sessions.pop(key, None)
        if sess is not None:
            sandbox.terminate(sess.proc)
            await self._reap(sess.proc)

    async def evict_idle(self, max_idle_s: float) -> int:
        now = time.time()
        victims: list[asyncio.subprocess.Process] = []
        async with self._lock:
            for key, sess in list(self._sessions.items()):
                if now - sess.last_used > max_idle_s or sess.proc.returncode is not None:
                    victims.append(sess.proc)
                    self._sessions.pop(key, None)
        for proc in victims:
            sandbox.terminate(proc)
        await asyncio.gather(*(self._reap(p) for p in victims), return_exceptions=True)
        return len(victims)

    async def shutdown(self) -> None:
        async with self._lock:
            procs = [s.proc for s in self._sessions.values()]
            self._sessions.clear()
        for proc in procs:
            sandbox.terminate(proc)
        await asyncio.gather(*(self._reap(p) for p in procs), return_exceptions=True)


# Process-wide manager (used by services.run_repl when repl_stateful is on).
MANAGER = SessionManager()


async def eviction_loop() -> None:
    """Background sweep evicting idle sessions; cancelled at shutdown."""
    while True:
        await asyncio.sleep(settings.repl_evict_interval_s)
        try:
            await MANAGER.evict_idle(settings.repl_idle_timeout_s)
        except Exception:
            pass
