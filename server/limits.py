"""Hardening primitives (plan Phase 7): a global concurrency cap, a
per-client request rate limiter, and a per-workspace disk-quota check.

All in-memory and process-local — fine for a single backend instance. A
multi-instance deployment would move these to Redis (see plan §7).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import HTTPException, Request, status

from config import settings

# --- Global concurrency cap ------------------------------------------------
_semaphore: asyncio.Semaphore | None = None


def job_semaphore() -> asyncio.Semaphore:
    """Process-wide cap on concurrent matlabc child processes."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))
    return _semaphore


# --- Per-client rate limiting (sliding 60s window) -------------------------
_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "anon"


async def rate_limit(request: Request) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    key = _client_key(request)
    now = time.monotonic()
    dq = _hits[key]
    while dq and now - dq[0] > 60.0:
        dq.popleft()
    if len(dq) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded; slow down",
            headers={"Retry-After": "60"},
        )
    dq.append(now)


# --- Disk quota ------------------------------------------------------------
def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def would_exceed_quota(workspace: Path, incoming_bytes: int) -> bool:
    quota = settings.user_quota_mb * 1024 * 1024
    if quota <= 0:
        return False
    return dir_size(workspace) + incoming_bytes > quota
