"""Stateful REPL sessions + a warm pool (plan §11 / Phase 7).

Keeps a long-lived ``matlabc -repl`` child per (user, session) so workspace
variables persist across ``/v1/repl`` calls. ``matlabc -repl`` prints no
prompt over a pipe, so each turn is delimited by appending a unique
``disp('<marker>')`` and reading stdout until the marker. stderr is merged
into stdout (``stderr=STDOUT``) so a turn's errors are captured inline.

Warm pool: matlabc has no ``cd`` builtin, so a pre-spawned worker can't be
retargeted to a session's deterministic workspace. Instead a pooled worker
**adopts its own pool dir as the session workspace** — files staged in the
deterministic dir are migrated in on adoption, and migrated back out (so they
stay reachable via /v1/files) when the session is retired. The pool amortises
matlabc's JIT cold-start: workers are pre-spawned and warmed with a no-op turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import limits
import sandbox
from config import settings
from workspaces import new_artifacts, snapshot, workspace_for


@dataclass
class ReplSession:
    proc: asyncio.subprocess.Process
    workspace: Path   # the dir the child actually runs in (pool dir or det)
    det: Path         # deterministic dir for this (user, session)
    lock: asyncio.Lock
    last_used: float


@dataclass
class Worker:
    proc: asyncio.subprocess.Process
    workdir: Path


def _migrate(src: Path, dst: Path) -> None:
    """Move top-level entries from src into dst (skip name clashes)."""
    try:
        if not src.exists() or src.resolve() == dst.resolve():
            return
    except OSError:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in list(src.iterdir()):
        target = dst / entry.name
        if target.exists():
            continue
        with contextlib.suppress(OSError):
            shutil.move(str(entry), str(target))


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


async def _reap(proc: asyncio.subprocess.Process) -> None:
    # Wait within the loop so the subprocess transport closes cleanly.
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


async def _spawn_repl(workdir: Path) -> asyncio.subprocess.Process:
    return await sandbox.spawn(
        [str(settings.matlabc_path), "-repl"],
        cwd=workdir,
        stderr=asyncio.subprocess.STDOUT,
    )


async def _warm(proc: asyncio.subprocess.Process, timeout: float = 30.0) -> bool:
    """Run a no-op turn to force JIT init and confirm responsiveness."""
    marker = f"<<<MLBC_WARM_{uuid.uuid4().hex}>>>"
    try:
        proc.stdin.write(("1+1;\n" + f"disp('{marker}')\n").encode("utf-8"))
        await proc.stdin.drain()
        _, timed_out = await _read_until(proc.stdout, marker.encode("utf-8"), timeout)
        return (not timed_out) and proc.returncode is None
    except Exception:
        return False


class WarmPool:
    def __init__(self) -> None:
        self._ready: list[Worker] = []
        self._lock = asyncio.Lock()
        self._fill_task: asyncio.Task | None = None

    @property
    def size(self) -> int:
        return len(self._ready)

    async def _make(self) -> Worker | None:
        if not settings.matlabc_path.exists():
            return None
        root = settings.workspace_root_path / ".pool"
        root.mkdir(parents=True, exist_ok=True)
        workdir = root / uuid.uuid4().hex[:16]
        workdir.mkdir(parents=True, exist_ok=True)
        proc = await _spawn_repl(workdir)
        if not await _warm(proc):
            sandbox.terminate(proc)
            await _reap(proc)
            shutil.rmtree(workdir, ignore_errors=True)
            return None
        return Worker(proc=proc, workdir=workdir)

    async def fill(self, target: int) -> None:
        async with self._lock:
            need = max(0, target - len(self._ready))
        if need <= 0:
            return
        made = await asyncio.gather(*[self._make() for _ in range(need)], return_exceptions=True)
        async with self._lock:
            for w in made:
                if isinstance(w, Worker):
                    self._ready.append(w)

    def request_fill(self, target: int) -> None:
        if self._fill_task is None or self._fill_task.done():
            self._fill_task = asyncio.create_task(self.fill(target))

    async def acquire(self) -> Worker | None:
        async with self._lock:
            return self._ready.pop() if self._ready else None

    async def shutdown(self) -> None:
        if self._fill_task is not None:
            self._fill_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._fill_task
        async with self._lock:
            workers = self._ready[:]
            self._ready.clear()
        for w in workers:
            sandbox.terminate(w.proc)
        await asyncio.gather(*(_reap(w.proc) for w in workers), return_exceptions=True)
        for w in workers:
            shutil.rmtree(w.workdir, ignore_errors=True)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ReplSession] = {}
        self._lock = asyncio.Lock()
        self.pool = WarmPool()

    @staticmethod
    def _key(user_id: str | None, session_id: str | None) -> str:
        return f"{user_id or 'anon'}::{session_id or 'default'}"

    @property
    def active(self) -> int:
        return len(self._sessions)

    def workspace_of(self, user_id: str | None, session_id: str | None) -> Path | None:
        sess = self._sessions.get(self._key(user_id, session_id))
        return sess.workspace if sess is not None else None

    async def ensure_session(self, user_id: str | None, session_id: str | None) -> ReplSession:
        key = self._key(user_id, session_id)
        async with self._lock:
            sess = self._sessions.get(key)
            if sess is not None and sess.proc.returncode is None:
                return sess
            det = workspace_for(user_id, session_id)
            worker = await self.pool.acquire() if settings.warm_pool_size > 0 else None
            if worker is not None:
                ws = worker.workdir
                _migrate(det, ws)  # adopt pool dir; bring staged files along
                proc = worker.proc
                self.pool.request_fill(settings.warm_pool_size)
            else:
                ws = det
                proc = await _spawn_repl(ws)
            sess = ReplSession(proc=proc, workspace=ws, det=det, lock=asyncio.Lock(), last_used=time.time())
            self._sessions[key] = sess
            return sess

    async def run_turn(self, user_id: str | None, session_id: str | None, source: str, timeout: float) -> dict:
        key = self._key(user_id, session_id)
        sess = await self.ensure_session(user_id, session_id)
        async with sess.lock:
            before = snapshot(sess.workspace)
            marker = f"<<<MLBC_TURN_{uuid.uuid4().hex}>>>"
            payload = source.rstrip("\n") + "\n" + f"disp('{marker}')\n"
            async with limits.job_semaphore():
                sess.proc.stdin.write(payload.encode("utf-8"))
                await sess.proc.stdin.drain()
                out, timed_out = await _read_until(sess.proc.stdout, marker.encode("utf-8"), timeout)
            sess.last_used = time.time()
            artifacts = new_artifacts(sess.workspace, before)

        alive = sess.proc.returncode is None
        if timed_out or not alive:  # unknown stream state -> recycle
            await self._retire_key(key)
            alive = False
        return {
            "stdout": out,
            "timed_out": timed_out,
            "alive": alive,
            "artifacts": artifacts,
            "workspace": str(sess.workspace),
        }

    async def _retire(self, sess: ReplSession) -> None:
        sandbox.terminate(sess.proc)
        await _reap(sess.proc)
        # Move adopted-pool-dir files back to the deterministic dir so they
        # stay reachable via /v1/files, then drop the pool dir.
        with contextlib.suppress(OSError):
            if sess.workspace.resolve() != sess.det.resolve():
                _migrate(sess.workspace, sess.det)
                shutil.rmtree(sess.workspace, ignore_errors=True)

    async def _retire_key(self, key: str) -> None:
        async with self._lock:
            sess = self._sessions.pop(key, None)
        if sess is not None:
            await self._retire(sess)

    async def evict_idle(self, max_idle_s: float) -> int:
        now = time.time()
        victims: list[ReplSession] = []
        async with self._lock:
            for key, sess in list(self._sessions.items()):
                if now - sess.last_used > max_idle_s or sess.proc.returncode is not None:
                    victims.append(sess)
                    self._sessions.pop(key, None)
        for sess in victims:
            await self._retire(sess)
        return len(victims)

    async def shutdown(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for sess in sessions:
            await self._retire(sess)
        await self.pool.shutdown()


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


async def prewarm() -> None:
    if settings.warm_pool_size > 0:
        with contextlib.suppress(Exception):
            await MANAGER.pool.fill(settings.warm_pool_size)
