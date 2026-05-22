"""Parse matlabc diagnostics out of compiler stdout/stderr.

matlabc emits clang-style ``file:line:col: severity: message`` lines; we
also catch the bare ``severity: message`` form. Best-effort and lossless
(the raw stderr is always returned alongside the structured list).
"""

from __future__ import annotations

import re

from models import Diagnostic

# file.m:LINE:COL: error: message
_FULL = re.compile(
    r"^(?:(?P<file>[^\s:][^:]*):)?(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<sev>error|warning|note):\s*(?P<msg>.*)$"
)
# bare:  error: message
_BARE = re.compile(r"^\s*(?P<sev>error|warning|note):\s*(?P<msg>.*)$")


def parse_diagnostics(text: str) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line:
            continue
        m = _FULL.match(line)
        if m:
            out.append(
                Diagnostic(
                    severity=m.group("sev"),
                    message=m.group("msg").strip(),
                    file=m.group("file"),
                    line=int(m.group("line")),
                    col=int(m.group("col")),
                )
            )
            continue
        m = _BARE.match(line)
        if m:
            out.append(Diagnostic(severity=m.group("sev"), message=m.group("msg").strip()))
    return out
