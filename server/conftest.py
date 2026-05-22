"""Test fixtures.

A fake ``matlabc`` stub mimics the subset of the real CLI the backend
drives, so the suite runs with no LLVM build. Env is configured at import
time — before any server module reads ``Settings`` — to point at the stub
and an isolated workspace root.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Body of the fake matlabc. Behaviour keys off marker tokens in the source
# so individual tests can request OK / error / hang / artifact paths.
_FAKE_BODY = r'''
import json, os, sys, time

args = sys.argv[1:]
EMIT = {"-emit-python": "python", "-emit-typescript": "typescript",
        "-emit-c": "c", "-emit-cpp": "cpp", "-emit-systemverilog": "systemverilog"}
mode, target, infile = "check", None, None
for a in args:
    if a == "-repl":
        mode = "repl"
    elif a == "-dap":
        mode = "dap"
    elif a in EMIT:
        mode, target = "emit", EMIT[a]
    elif not a.startswith("-"):
        infile = a

if mode == "dap":
    rin, rout = sys.stdin.buffer, sys.stdout.buffer
    seq = [0]

    def read_frame():
        hdr = b""
        while not hdr.endswith(b"\r\n\r\n"):
            ch = rin.read(1)
            if not ch:
                return None
            hdr += ch
        length = 0
        for ln in hdr.split(b"\r\n"):
            if ln.lower().startswith(b"content-length:"):
                length = int(ln.split(b":", 1)[1].strip())
        body = b""
        while len(body) < length:
            c = rin.read(length - len(body))
            if not c:
                return None
            body += c
        return json.loads(body.decode("utf-8"))

    def send(obj):
        seq[0] += 1
        obj["seq"] = seq[0]
        b = json.dumps(obj).encode("utf-8")
        rout.write(b"Content-Length: %d\r\n\r\n" % len(b))
        rout.write(b)
        rout.flush()

    def respond(req, body=None):
        send({"type": "response", "request_seq": req["seq"], "success": True,
              "command": req["command"], "body": body or {}})

    def event(name, body=None):
        send({"type": "event", "event": name, "body": body or {}})

    while True:
        req = read_frame()
        if req is None:
            break
        cmd = req.get("command")
        if cmd == "initialize":
            respond(req, {"supportsConfigurationDoneRequest": True})
            event("initialized")
        elif cmd == "setBreakpoints":
            bps = req.get("arguments", {}).get("breakpoints", [])
            respond(req, {"breakpoints": [{"verified": True, "line": b.get("line", 1)} for b in bps]})
        elif cmd == "launch":
            respond(req)
            event("stopped", {"reason": "breakpoint", "threadId": 1, "allThreadsStopped": True})
        elif cmd == "threads":
            respond(req, {"threads": [{"id": 1, "name": "main"}]})
        elif cmd == "stackTrace":
            respond(req, {"stackFrames": [{"id": 1, "name": "main", "line": 1, "column": 1}], "totalFrames": 1})
        elif cmd == "scopes":
            respond(req, {"scopes": [{"name": "Locals", "variablesReference": 1000, "expensive": False}]})
        elif cmd == "variables":
            respond(req, {"variables": [{"name": "x", "value": "42", "variablesReference": 0}]})
        elif cmd in ("next", "stepIn", "stepOut"):
            respond(req)
            event("stopped", {"reason": "step", "threadId": 1, "allThreadsStopped": True})
        elif cmd == "continue":
            respond(req, {"allThreadsContinued": True})
            event("terminated")
            break
        elif cmd == "disconnect":
            respond(req)
            break
        else:
            respond(req)
    sys.exit(0)

if mode == "repl":
    src = sys.stdin.read()
    if "INFLOOP" in src:
        while True:
            time.sleep(0.2)
    if "PLOT" in src:
        with open(os.path.join(os.getcwd(), "figure_1.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
    if "ERR" in src:
        sys.stderr.write("error: simulated repl error\n")
        sys.exit(1)
    for line in src.splitlines():
        line = line.strip()
        if line:
            sys.stdout.write("ans = %s\n" % line)
    sys.exit(0)

src = open(infile).read() if infile else ""
if "ERR" in src:
    sys.stderr.write("%s:1:1: error: simulated syntax error\n" % infile)
    sys.exit(1)
if mode == "check":
    sys.exit(0)
if mode == "emit":
    out = {
        "python": "# generated python\nprint('hi')\n",
        "typescript": "// generated ts\nconsole.log('hi')\n",
        "c": "/* generated c */\nint main(void){return 0;}\n",
        "cpp": "// generated cpp\nint main(){return 0;}\n",
        "systemverilog": "// generated sv\nmodule m; endmodule\n",
    }[target]
    sys.stdout.write(out)
    sys.exit(0)
'''

_TMP = Path(tempfile.mkdtemp(prefix="mlb_test_"))
_FAKE = _TMP / "fake_matlabc"
# Shebang points at the *current* interpreter so it survives the sandbox's
# scrubbed PATH (anaconda/homebrew python is not on /usr/bin).
_FAKE.write_text(f"#!{sys.executable}\n" + _FAKE_BODY)
_FAKE.chmod(0o755)

os.environ["MATLAB_BACKEND_MATLABC_BIN"] = str(_FAKE)
os.environ["MATLAB_BACKEND_WORKSPACE_ROOT"] = str(_TMP / "ws")
os.environ["MATLAB_BACKEND_WALL_TIMEOUT_S"] = "2"

# Force the chat endpoint into offline (retrieval-only) mode and index a tiny
# hermetic docs corpus instead of the real repo docs. Empty OPENAI_API_KEY in
# os.environ overrides any value from server/.env.
os.environ["OPENAI_API_KEY"] = ""
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
