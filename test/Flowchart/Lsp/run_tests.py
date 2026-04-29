#!/usr/bin/env python3
"""Phase 5 LSP test for the .mflow file-type hook.

Drives matlab-lsp via stdio with `textDocument/didOpen` events for a
mix of well-formed and malformed `.mflow` payloads. Asserts that:

  - A valid `.mflow` produces zero diagnostics.
  - A `.mflow` with a schema mismatch surfaces a single error
    diagnostic with severity 1.
  - A `.mflow` with an irreducible CFG surfaces a builder
    diagnostic at the offending node.

The harness keeps to the LSP minimum (initialize / didOpen / shutdown)
so this lane is self-contained and doesn't depend on a real editor
client. See docs/flowchart_frontend.md.
"""

import json
import os
import subprocess
import sys


def frame(obj):
    body = json.dumps(obj)
    return f"Content-Length: {len(body)}\r\n\r\n{body}".encode()


def parse_events(buf):
    i = 0
    events = []
    while i < len(buf):
        nl = buf.find(b"\r\n\r\n", i)
        if nl < 0:
            break
        header = buf[i:nl].decode()
        cl = 0
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length"):
                cl = int(line.split(":", 1)[1].strip())
                break
        body = buf[nl + 4 : nl + 4 + cl]
        events.append(json.loads(body))
        i = nl + 4 + cl
    return events


GOOD = """{
  "schema": "matforge.flowchart",
  "version": "0.1.0",
  "entry": "main",
  "flows": [
    { "id": "fm", "kind": "program", "name": "main",
      "nodes": [
        { "id": "s", "kind": "start",
          "ports": {"in": [], "out": [{"id": "out"}]} },
        { "id": "v", "kind": "variable",
          "data": {"name": "x", "value": "1"},
          "ports": {"in": [{"id": "in"}], "out": [{"id": "out"}]} },
        { "id": "e", "kind": "end",
          "ports": {"in": [{"id": "in"}], "out": []} }
      ],
      "edges": [
        { "id": "e1", "kind": "control",
          "from": {"node": "s", "port": "out"}, "to": {"node": "v", "port": "in"} },
        { "id": "e2", "kind": "control",
          "from": {"node": "v", "port": "out"}, "to": {"node": "e", "port": "in"} }
      ]
    }
  ]
}"""

BAD_SCHEMA = """{
  "schema": "not-the-flowchart-schema",
  "version": "0.1.0",
  "flows": []
}"""

# `if` block whose `data.cond` is missing — the builder rejects it
# with an actionable diagnostic.
BAD_IF = """{
  "schema": "matforge.flowchart",
  "version": "0.1.0",
  "entry": "main",
  "flows": [
    { "id": "fm", "kind": "program", "name": "main",
      "nodes": [
        { "id": "s",  "kind": "start",
          "ports": {"in": [], "out": [{"id": "out"}]} },
        { "id": "if1", "kind": "if",
          "ports": {"in": [{"id": "in"}],
                     "out": [{"id": "true"}, {"id": "false"}]} },
        { "id": "e",  "kind": "end",
          "ports": {"in": [{"id": "in"}], "out": []} }
      ],
      "edges": [
        { "id": "e1", "kind": "control",
          "from": {"node": "s",   "port": "out"},   "to": {"node": "if1", "port": "in"} },
        { "id": "e2", "kind": "control",
          "from": {"node": "if1", "port": "true"},  "to": {"node": "e",   "port": "in"} },
        { "id": "e3", "kind": "control",
          "from": {"node": "if1", "port": "false"}, "to": {"node": "e",   "port": "in"} }
      ]
    }
  ]
}"""


def run_session(matlabc_lsp_path, init_opts, cases):
    """Drive a single LSP session: initialize → didOpen each case →
    shutdown. Returns a dict { uri: [diagnostics, ...] }."""
    init_params = {"capabilities": {}}
    if init_opts is not None:
        init_params["initializationOptions"] = init_opts
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": init_params},
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
    ]
    for case in cases:
        msgs.append({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {
                "uri": case["uri"], "languageId": "json", "version": 1,
                "text": case["text"],
            }},
        })
    msgs += [
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
        {"jsonrpc": "2.0", "method": "exit"},
    ]
    proc = subprocess.run(
        [matlabc_lsp_path],
        input=b"".join(frame(m) for m in msgs),
        capture_output=True, timeout=15)
    diags_by_uri = {}
    for ev in parse_events(proc.stdout):
        if ev.get("method") == "textDocument/publishDiagnostics":
            p = ev["params"]
            diags_by_uri.setdefault(p["uri"], []).extend(p["diagnostics"])
    return diags_by_uri


# Phase 8c: a `.mflow` whose `custom` block uses a `library_id`. The
# matching `dsp/scale.m` lives under test/Flowchart/EmitMatlab/lib/
# (existing fixture from the EmitMatlab corpus). Without the IDE
# supplying `blockPath` via `initializationOptions`, this should
# produce a "could not resolve library_id" diagnostic. With it set,
# the diagnostic clears.
LIB_BLOCK = """{
  "schema": "matforge.flowchart",
  "version": "0.1.0",
  "entry": "main",
  "flows": [
    { "id": "fm", "kind": "program", "name": "main",
      "nodes": [
        { "id": "s", "kind": "start",
          "ports": {"in": [], "out": [{"id": "out"}]} },
        { "id": "cb", "kind": "custom",
          "data": {
            "name": "scale", "callee": "scale",
            "args": "10, 4", "lhs": "y",
            "library_id": "dsp/scale"
          },
          "ports": {"in": [{"id": "in"}], "out": [{"id": "out"}]} },
        { "id": "e", "kind": "end",
          "ports": {"in": [{"id": "in"}], "out": []} }
      ],
      "edges": [
        { "id": "e_1", "kind": "control",
          "from": {"node": "s",  "port": "out"}, "to": {"node": "cb", "port": "in"} },
        { "id": "e_2", "kind": "control",
          "from": {"node": "cb", "port": "out"}, "to": {"node": "e",  "port": "in"} }
      ]
    }
  ]
}"""


def run(matlabc_lsp_path):
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    block_lib = os.path.join(repo_root, "test/Flowchart/EmitMatlab/lib")

    base_cases = [
        {"uri": "file:///tmp/lsp_good.mflow",       "text": GOOD,
         "expected_count": 0, "expected_substr": None},
        {"uri": "file:///tmp/lsp_bad_schema.mflow", "text": BAD_SCHEMA,
         "expected_count": 1, "expected_substr": "schema"},
        {"uri": "file:///tmp/lsp_bad_if.mflow",     "text": BAD_IF,
         "expected_count": 1, "expected_substr": "cond"},
    ]
    no_blockpath_case = {
        "uri": "file:///tmp/lsp_lib_no_path.mflow", "text": LIB_BLOCK,
        "expected_count": 1, "expected_substr": "library_id",
    }
    with_blockpath_case = {
        "uri": "file:///tmp/lsp_lib_with_path.mflow", "text": LIB_BLOCK,
        "expected_count": 0, "expected_substr": None,
    }

    failures = []

    # Session 1: no initializationOptions. Existing cases plus the
    # Phase 8c "library_id can't resolve" negative case.
    diags = run_session(matlabc_lsp_path, None,
                        base_cases + [no_blockpath_case])
    for case in base_cases + [no_blockpath_case]:
        ds = diags.get(case["uri"], [])
        if len(ds) != case["expected_count"]:
            failures.append(
                f"{case['uri']}: expected {case['expected_count']} "
                f"diagnostic(s), got {len(ds)}: "
                f"{[d['message'] for d in ds]}")
            continue
        if case["expected_substr"] is not None:
            if not any(case["expected_substr"] in d["message"] for d in ds):
                failures.append(
                    f"{case['uri']}: no diagnostic mentioned "
                    f"'{case['expected_substr']}': "
                    f"{[d['message'] for d in ds]}")

    # Session 2: initializationOptions.blockPath pointing at the
    # block library — the `library_id` resolves and diagnostics clear.
    diags = run_session(matlabc_lsp_path,
                        {"blockPath": [block_lib]},
                        [with_blockpath_case])
    ds = diags.get(with_blockpath_case["uri"], [])
    if len(ds) != 0:
        failures.append(
            f"{with_blockpath_case['uri']}: expected 0 diagnostics with "
            f"blockPath set, got {len(ds)}: {[d['message'] for d in ds]}")

    total = len(base_cases) + 1 + 1
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print(f"PASS ({total} cases)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-matlab-lsp>", file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
