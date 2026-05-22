"""Live-server integration fixtures.

Boots a real ``uvicorn`` subprocess and exercises it over HTTP/WS — distinct
from the in-process TestClient unit tests. Uses ``build/matlabc`` when it
exists (the real compiler), otherwise a fake stub, so it runs anywhere.

Run with ``just backend-itest`` (not part of the default ``pytest`` run).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _matlabc(tmp_path_factory) -> tuple[str, bool]:
    real = REPO_ROOT / "build" / "matlabc"
    if real.exists():
        return str(real), True
    from _devfake import write_fake

    return str(write_fake(tmp_path_factory.mktemp("fakebin"))), False


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    port = _free_port()
    ws_root = tmp_path_factory.mktemp("itest_ws")
    matlabc, is_real = _matlabc(tmp_path_factory)
    env = {
        **os.environ,
        "MATLAB_BACKEND_MATLABC_BIN": matlabc,
        "MATLAB_BACKEND_WORKSPACE_ROOT": str(ws_root),
        "MATLAB_BACKEND_WARM_POOL_SIZE": "1",     # exercise the pool live
        "MATLAB_BACKEND_WALL_TIMEOUT_S": "15",
        "MATLAB_BACKEND_SOURCE_CONTEXT_ROOT": str(REPO_ROOT),
        "OPENAI_API_KEY": "",                      # offline, deterministic chat
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(SERVER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise RuntimeError(f"server exited early (rc={proc.returncode}):\n{out}")
        try:
            if httpx.get(base + "/healthz", timeout=1.0).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("server did not become healthy in time")

    yield {"base": base, "real": is_real}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def base_url(server) -> str:
    return server["base"]


@pytest.fixture(scope="session")
def is_real(server) -> bool:
    return server["real"]


@pytest.fixture()
def http(base_url):
    with httpx.Client(base_url=base_url, timeout=30.0) as c:
        yield c
