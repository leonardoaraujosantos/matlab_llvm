"""Tier-2 syscall sandbox (plan Phase 7).

Wraps a child argv in an OS sandbox (bubblewrap / firejail / nsjail) for
isolation beyond rlimits: read-only root, the workspace bind-mounted
read-write, no network (by default), dropped capabilities. Linux-only — on
macOS or when the tool is missing it transparently falls back to the bare
argv (rlimit-only). The real boundary in production is the container; this
is defense-in-depth (see plan §7).

The argv builders are unit-tested; the actual jailing is validated in the
deployed container.

**Startup self-test** (``probe()``): before declaring a backend active, we
run a no-op invocation. If the host blocks unprivileged user namespaces
(``kernel.unprivileged_userns_clone=0``, common on locked-down Docker
hosts), bwrap fails with "No permissions to create new namespace" — we
catch that, downgrade the runtime backend to ``none``, and surface the
reason on ``/healthz`` so ops can see *why* tier-2 didn't engage.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config import settings

log = logging.getLogger("matlab_backend.jail")

BACKENDS = ("none", "bwrap", "firejail", "nsjail")
_TOOL = {"bwrap": "bwrap", "firejail": "firejail", "nsjail": "nsjail"}
_warned: set[str] = set()
_probe_reason: str | None = None  # populated by probe() when a backend is unusable


def _bwrap(argv: list[str], ws: str, allow_net: bool) -> list[str]:
    cmd = [
        "bwrap",
        "--ro-bind", "/", "/",      # read-only host root
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", ws, ws,            # workspace read-write
        "--chdir", ws,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if not allow_net:
        cmd.append("--unshare-net")
    cmd.append("--")
    return cmd + argv


def _firejail(argv: list[str], ws: str, allow_net: bool) -> list[str]:
    cmd = [
        "firejail",
        "--quiet",
        "--noprofile",
        "--caps.drop=all",
        "--nonewprivs",
        "--nogroups",
        f"--whitelist={ws}",
    ]
    if not allow_net:
        cmd.append("--net=none")
    cmd.append("--")
    return cmd + argv


def _nsjail(argv: list[str], ws: str, allow_net: bool) -> list[str]:
    cmd = [
        "nsjail",
        "--quiet",
        "--mode", "o",            # run once
        "--chroot", "/",
        "--cwd", ws,
        "--bindmount", f"{ws}:{ws}",
        "--disable_clone_newuser",
    ]
    if not allow_net:
        cmd.append("--disable_clone_newnet")
    cmd.append("--")
    return cmd + argv


_BUILDERS = {"bwrap": _bwrap, "firejail": _firejail, "nsjail": _nsjail}


def wrap(argv: list[str], workspace: str | Path) -> list[str]:
    """Return ``argv`` wrapped in the configured sandbox, or unchanged."""
    backend = settings.sandbox_backend
    if backend == "none":
        return argv
    if backend not in _BUILDERS:
        if backend not in _warned:
            log.warning("unknown sandbox_backend %r; running rlimit-only", backend)
            _warned.add(backend)
        return argv
    if shutil.which(_TOOL[backend]) is None:
        if backend not in _warned:
            log.warning("sandbox tool %r not found; running rlimit-only", _TOOL[backend])
            _warned.add(backend)
        return argv
    return _BUILDERS[backend](argv, str(workspace), settings.sandbox_allow_net)


def probe() -> tuple[bool, str | None]:
    """Self-test the configured backend with a trivial no-op invocation.

    Returns ``(works, reason)``. When ``works`` is False, ``reason`` is the
    short message to surface on ``/healthz`` and the caller should fall back
    to ``none`` for actual matlabc invocations.

    Treats configured-as-``none`` and tool-missing-on-PATH as "works=False,
    reason=None" — those are normal silent fallbacks, not error states.
    """
    global _probe_reason
    backend = settings.sandbox_backend
    if backend == "none":
        _probe_reason = None
        return False, None
    if backend not in _BUILDERS or shutil.which(_TOOL[backend]) is None:
        _probe_reason = None
        return False, None
    # Probe by running a no-op argv (``true``) through the full wrapper.
    # Use the actual workspace root so any host-level mount restrictions
    # show up here too.
    ws = str(settings.workspace_root_path)
    Path(ws).mkdir(parents=True, exist_ok=True)
    wrapped = _BUILDERS[backend]([shutil.which("true") or "/bin/true"], ws, settings.sandbox_allow_net)
    try:
        cp = subprocess.run(wrapped, capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        _probe_reason = f"sandbox probe failed: {e}"
        log.warning(_probe_reason)
        return False, _probe_reason
    if cp.returncode == 0:
        _probe_reason = None
        return True, None
    # Surface a short, actionable hint instead of the full stderr.
    err = (cp.stderr or b"").decode("utf-8", "replace").strip()
    if "user namespace" in err.lower() or "new namespace" in err.lower():
        _probe_reason = "host blocks unprivileged user namespaces — set kernel.unprivileged_userns_clone=1 or grant CAP_SYS_ADMIN"
    elif "permission" in err.lower():
        _probe_reason = "permission denied — container lacks required capabilities"
    else:
        _probe_reason = f"sandbox probe exited {cp.returncode}: {err[:160]}"
    log.warning("tier-2 sandbox %r unusable: %s", backend, _probe_reason)
    return False, _probe_reason


def probe_reason() -> str | None:
    return _probe_reason
