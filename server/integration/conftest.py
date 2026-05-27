"""Live-server integration fixtures.

Two modes, picked by env vars:

* **Local boot** (default) — spawns a ``uvicorn`` subprocess in this repo,
  pointing at ``build/matlabc`` when present, else a fake stub. Used by
  ``just backend-itest``.
* **Remote** — set ``BACKEND_URL=https://…`` to skip the subprocess and run
  the same tests against an already-deployed backend. Auth is supplied
  either via ``BACKEND_TOKEN=<bearer>`` (a static / pre-minted token) or
  via ``CYBERDYNE_USER`` + ``CYBERDYNE_PASS`` (logs into the auth server
  configured at ``CYBERDYNE_AUTH_URL``, default
  ``https://auth.backend.coolify.cyberdynecorp.ai``). Used by
  ``just backend-test-remote``.

The fixture set is identical in both modes: ``base_url``, ``http`` (httpx
client with bearer pre-injected), ``is_real`` (matlabc present + plot
supported?), and ``ws_auth_qs`` (query-string token for the DAP WebSocket,
which can't carry an ``Authorization`` header).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parent

DEFAULT_CYBERDYNE_AUTH = "https://auth.backend.coolify.cyberdynecorp.ai"


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


def _cyberdyne_login(auth_url: str, email: str, password: str) -> str:
    url = auth_url.rstrip("/") + "/api/v1/auth/login"
    r = httpx.post(url, json={"email": email, "password": password}, timeout=15.0)
    r.raise_for_status()
    body = r.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"login at {url} returned no access_token: {body}")
    return token


def _wait_healthy(base_url: str, token: str | None, deadline_s: float = 60.0, proc=None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise RuntimeError(f"server exited early (rc={proc.returncode}):\n{out}")
        try:
            # /healthz is unauthenticated; if it ever changes, this still works
            # without the header.
            if httpx.get(base_url + "/healthz", headers=headers, timeout=5.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"backend at {base_url} did not become healthy in {deadline_s:.0f}s")


@pytest.fixture(scope="session")
def backend(tmp_path_factory):
    """Resolve a backend to test against — remote URL or a freshly booted local server."""
    remote_url = os.environ.get("BACKEND_URL", "").strip().rstrip("/")
    if remote_url:
        token = os.environ.get("BACKEND_TOKEN", "").strip()
        if not token:
            email = os.environ.get("CYBERDYNE_USER", "").strip()
            pw = os.environ.get("CYBERDYNE_PASS", "").strip()
            if email and pw:
                token = _cyberdyne_login(
                    os.environ.get("CYBERDYNE_AUTH_URL", DEFAULT_CYBERDYNE_AUTH),
                    email,
                    pw,
                )
        _wait_healthy(remote_url, token)
        # Probe plot support for the optional plot test.
        try:
            r = httpx.post(
                remote_url + "/v1/plot?raw=true",
                headers={"Authorization": f"Bearer {token}"} if token else {},
                json={"source": "plot([1 2 3])", "session_id": "probe_plot_support"},
                timeout=60.0,
            )
            plot_ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/")
        except Exception:
            plot_ok = False
        yield {"base": remote_url, "real": True, "plot": plot_ok, "token": token, "mode": "remote"}
        return

    # Local boot path.
    port = _free_port()
    ws_root = tmp_path_factory.mktemp("itest_ws")
    matlabc, is_real = _matlabc(tmp_path_factory)
    env = {
        **os.environ,
        "MATLAB_BACKEND_MATLABC_BIN": matlabc,
        "MATLAB_BACKEND_WORKSPACE_ROOT": str(ws_root),
        "MATLAB_BACKEND_WARM_POOL_SIZE": "1",
        "MATLAB_BACKEND_WALL_TIMEOUT_S": "15",
        "MATLAB_BACKEND_SOURCE_CONTEXT_ROOT": str(REPO_ROOT),
        "OPENAI_API_KEY": "",  # offline, deterministic chat
        "MATLAB_BACKEND_MCP_REQUIRE_AUTH": "1",
        "MATLAB_BACKEND_MCP_TOKEN_SECRET": "itest-mcp-secret",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(SERVER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(base, token=None, deadline_s=60.0, proc=proc)
    except Exception:
        proc.terminate()
        raise

    # Plot only works when matlabc was built with -DMATLAB_LLVM_WITH_PLOT.
    plot_ok = is_real
    if plot_ok:
        try:
            r = httpx.post(base + "/v1/plot?raw=true", json={"source": "plot([1 2 3])", "session_id": "probe_plot_support"}, timeout=30.0)
            plot_ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/")
        except Exception:
            plot_ok = False

    yield {"base": base, "real": is_real, "plot": plot_ok, "token": None, "mode": "local"}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def base_url(backend) -> str:
    return backend["base"]


@pytest.fixture(scope="session")
def is_real(backend) -> bool:
    return backend["real"]


@pytest.fixture(scope="session")
def plot_supported(backend) -> bool:
    return backend["plot"]


@pytest.fixture(scope="session")
def auth_token(backend) -> str | None:
    return backend["token"]


@pytest.fixture(scope="session")
def mode(backend) -> str:
    return backend["mode"]


@pytest.fixture()
def http(base_url, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as c:
        yield c


@pytest.fixture()
def ws_auth_qs(auth_token) -> str:
    """Query-string fragment for WS auth — ``?token=…&`` or empty."""
    if not auth_token:
        return ""
    return urlencode({"token": auth_token}) + "&"
