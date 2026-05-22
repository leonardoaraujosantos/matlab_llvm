"""Test fixtures.

A fake ``matlabc`` stub mimics the subset of the real CLI the backend
drives, so the suite runs with no LLVM build. Env is configured at import
time — before any server module reads ``Settings`` — to point at the stub
and an isolated workspace root.
"""

import os
import tempfile
from pathlib import Path

import pytest

from _devfake import write_fake

_TMP = Path(tempfile.mkdtemp(prefix="mlb_test_"))
_FAKE = write_fake(_TMP)

os.environ["MATLAB_BACKEND_MATLABC_BIN"] = str(_FAKE)
os.environ["MATLAB_BACKEND_WORKSPACE_ROOT"] = str(_TMP / "ws")
os.environ["MATLAB_BACKEND_WALL_TIMEOUT_S"] = "2"
# Disable the warm pool for the API-level (TestClient) tests: each TestClient
# spins a fresh event loop, and a global-pool worker prewarmed in one loop
# can't be reaped in another. The pool itself is covered by the local-manager
# unit tests in test_sessions.py.
os.environ["MATLAB_BACKEND_WARM_POOL_SIZE"] = "0"

# Force the chat endpoint into offline (retrieval-only) mode and index a tiny
# hermetic docs corpus instead of the real repo docs. Empty OPENAI_API_KEY in
# os.environ overrides any value from server/.env.
os.environ["OPENAI_API_KEY"] = ""
# Default tests run in open auth mode; clear any inherited auth config.
os.environ.pop("CYBERDYNE_AUTH_URL", None)
os.environ.pop("MATLAB_BACKEND_API_TOKEN", None)
_SRC_CTX = _TMP / "src_ctx"
_DOCS = _SRC_CTX / "docs"
_DOCS.mkdir(parents=True, exist_ok=True)
(_DOCS / "optim.md").write_text(
    "# Optimization Toolbox\n\n## fmincon\n"
    "fmincon finds the minimum of a constrained nonlinear multivariable "
    "function. Use it for constrained optimization with bounds and "
    "nonlinear constraints.\n"
)
(_DOCS / "emit_sv.md").write_text(
    "# SystemVerilog emission\n\n## -emit-sv\n"
    "The -emit-systemverilog (-emit-sv) flag emits synthesizable "
    "SystemVerilog from a .m file for ASIC/FPGA targets.\n"
)
os.environ["MATLAB_BACKEND_SOURCE_CONTEXT_ROOT"] = str(_SRC_CTX)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_global_manager():
    """Drop any global-MANAGER state a test leaked, synchronously (killpg,
    no await), so a child proc bound to one test's event loop is never
    awaited in another's."""
    yield
    import sandbox
    import sessions

    mgr = sessions.MANAGER
    for sess in list(mgr._sessions.values()):
        sandbox.terminate(sess.proc)
    mgr._sessions.clear()
    for worker in list(mgr.pool._ready):
        sandbox.terminate(worker.proc)
    mgr.pool._ready.clear()
    mgr.pool._fill_task = None
