"""Tier-2 syscall sandbox (plan Phase 7).

Wraps a child argv in an OS sandbox (bubblewrap / firejail / nsjail) for
isolation beyond rlimits: read-only root, the workspace bind-mounted
read-write, no network (by default), dropped capabilities. Linux-only — on
macOS or when the tool is missing it transparently falls back to the bare
argv (rlimit-only). The real boundary in production is the container; this
is defense-in-depth (see plan §7).

The argv builders are unit-tested; the actual jailing is validated in the
deployed container.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config import settings

log = logging.getLogger("matlab_backend.jail")

BACKENDS = ("none", "bwrap", "firejail", "nsjail")
_TOOL = {"bwrap": "bwrap", "firejail": "firejail", "nsjail": "nsjail"}
_warned: set[str] = set()


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
