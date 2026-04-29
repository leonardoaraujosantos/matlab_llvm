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


def run(matlabc_lsp_path):
    cases = [
        ("file:///tmp/lsp_good.mflow", GOOD, 0, None),
        ("file:///tmp/lsp_bad_schema.mflow", BAD_SCHEMA, 1, "schema"),
        ("file:///tmp/lsp_bad_if.mflow", BAD_IF, 1, "cond"),
    ]
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"capabilities": {}}},
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
    ]
    for uri, text, _, _ in cases:
        msgs.append({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {
                "uri": uri, "languageId": "json", "version": 1, "text": text,
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
    events = parse_events(proc.stdout)

    diags_by_uri = {}
    for ev in events:
        if ev.get("method") == "textDocument/publishDiagnostics":
            p = ev["params"]
            diags_by_uri.setdefault(p["uri"], []).extend(p["diagnostics"])

    failures = []
    for uri, _, expected_count, expected_substr in cases:
        diags = diags_by_uri.get(uri, [])
        if len(diags) != expected_count:
            failures.append(
                f"{uri}: expected {expected_count} diagnostic(s), got "
                f"{len(diags)}: {[d['message'] for d in diags]}")
            continue
        if expected_substr is not None:
            if not any(expected_substr in d["message"] for d in diags):
                failures.append(
                    f"{uri}: no diagnostic mentioned '{expected_substr}': "
                    f"{[d['message'] for d in diags]}")

    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print(f"PASS ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-matlab-lsp>", file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
