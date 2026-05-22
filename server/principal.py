"""Request-scoped authenticated principal.

Set by the auth layer (the verified identity id) and read by workspace
resolution so workspaces are isolated *per identity* rather than by a
client-supplied ``user_id`` (which a caller could spoof). Uses a ContextVar,
so each request task carries its own value.
"""

from __future__ import annotations

import contextvars

_principal: contextvars.ContextVar[str | None] = contextvars.ContextVar("principal", default=None)


def set_principal(pid: str | None) -> None:
    _principal.set(pid)


def get_principal() -> str | None:
    return _principal.get()


def effective_user(user_id: str | None) -> str | None:
    """The authenticated principal wins over a client-supplied user_id; falls
    back to the client value when unauthenticated (token/none modes)."""
    return get_principal() or user_id
