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
import os, sys, time

args = sys.argv[1:]
EMIT = {"-emit-python": "python", "-emit-typescript": "typescript",
        "-emit-c": "c", "-emit-cpp": "cpp", "-emit-systemverilog": "systemverilog"}
mode, target, infile = "check", None, None
for a in args:
    if a == "-repl":
        mode = "repl"
    elif a in EMIT:
        mode, target = "emit", EMIT[a]
    elif not a.startswith("-"):
        infile = a

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


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as c:
        yield c
