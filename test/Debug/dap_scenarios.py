"""DAP scenarios for matlabc.

Each scenario function takes (matlabc_path, program_path) and raises
on failure. The runner picks them up by their `scn_` prefix.

Scenarios reuse `dap_program.m`. Line numbers reference that file:

    1   % comment header
    2   % comment header
    3   % comment header
    4   x = 10;
    5   y = 20;
    6   z = x + y;
    7   (blank)
    8   for i = 1:3
    9       z = z + i;
    10  end
    11  (blank)
    12  disp(z);

Hook lines emitted (per extract_hook_lines.py): 4, 5, 6, 8, 9, 12. The
loop-body hook on line 9 fires three times at runtime.
"""

import os

from dap_client import DapClient, DapError, initialize_and_launch


def _stop_event(client, timeout=5.0):
    ev = client.wait_event("stopped", timeout=timeout)
    body = ev.get("body") or {}
    return body


def _vars_by_name(client, ref=1):
    body = client.request("variables", {"variablesReference": ref})
    out = {}
    for v in body.get("variables") or []:
        out[v.get("name")] = v.get("value")
    return out


def _abs(p):
    return os.path.realpath(p)


# --- scenarios ---------------------------------------------------------------

def scn_basic_breakpoint(matlabc, program):
    """Plain breakpoint: stop with reason='breakpoint', resume to end."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(
            c,
            stop_on_entry=False,
            breakpoints=[{"line": 5}],
        )
        body = _stop_event(c)
        assert body.get("reason") == "breakpoint", \
            f"expected reason=breakpoint, got {body!r}"
        assert body.get("line") == 5, \
            f"expected line=5, got {body!r}"
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_step_reason(matlabc, program):
    """Stepping must surface as reason='step', not 'breakpoint'.

    This is the regression that motivated the fix in tools/matlabc/
    main.cpp: every stop was hardcoded as 'breakpoint' regardless of
    whether a bp matched. Steps come through with BpIdx == -1; the
    server must translate that to 'step'.
    """
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, stop_on_entry=True)
        body = _stop_event(c)
        assert body.get("reason") == "step", \
            f"first stop reason should be 'step', got {body!r}"
        assert body.get("line") == 4, \
            f"first stop should be at line 4, got {body!r}"
        c.request("next")
        body = _stop_event(c)
        assert body.get("reason") == "step", \
            f"step-over stop reason should be 'step', got {body!r}"
        assert body.get("line") == 5, \
            f"second stop should be at line 5, got {body!r}"
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_stack_scope_variables(matlabc, program):
    """stackTrace / scopes / variables introspection while paused."""
    with DapClient(matlabc, program) as c:
        # bp at line 8 — by then x=10, y=20, z=30 have all been assigned.
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        body = _stop_event(c)
        assert body.get("line") == 8, body

        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        assert frames, f"stackTrace empty: {st!r}"
        assert frames[0].get("line") == 8, \
            f"top frame should be at line 8, got {frames[0]!r}"

        sc = c.request("scopes", {"frameId": frames[0].get("id", 0)})
        scopes = sc.get("scopes") or []
        assert scopes, f"no scopes returned: {sc!r}"
        locals_scope = scopes[0]
        assert locals_scope.get("name") == "Locals", \
            f"expected Locals scope, got {locals_scope!r}"
        ref = locals_scope.get("variablesReference")
        assert ref, f"variablesReference missing: {locals_scope!r}"

        vars_ = _vars_by_name(c, ref=ref)
        # Workspace must show the three scalars assigned on lines 4-6.
        for name, expected in (("x", "10"), ("y", "20"), ("z", "30")):
            assert vars_.get(name) == expected, \
                f"expected {name}={expected}, got vars={vars_!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_set_variable(matlabc, program):
    """setVariable mutates the workspace; subsequent variables read it back."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)
        before = _vars_by_name(c)
        assert before.get("x") == "10", f"sanity: x should be 10, got {before!r}"

        resp = c.request("setVariable", {
            "variablesReference": 1,
            "name": "x",
            "value": "99",
        })
        assert resp.get("value") == "99", f"setVariable resp: {resp!r}"

        after = _vars_by_name(c)
        assert after.get("x") == "99", \
            f"x should read back as 99, got {after!r}"
        # y / z untouched.
        assert after.get("y") == "20", after
        assert after.get("z") == "30", after

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_conditional_breakpoint(matlabc, program):
    """Conditional bp: false conditions silently resume, true ones stop.

    Two bps are set in the same request. Line 5's condition is false at
    fire time so the server must suppress that pause; line 6's is true
    so the server must surface a 'stopped' event there. Both conditions
    reference `x`, which is script-scope and therefore lives in the
    workspace under ReplMode — the path the conditional evaluator
    actually exercises (see main.cpp:1607-1611).
    """
    with DapClient(matlabc, program) as c:
        initialize_and_launch(
            c,
            breakpoints=[
                {"line": 5, "condition": "x == 999"},  # always false
                {"line": 6, "condition": "x == 10"},   # always true
            ],
        )
        body = _stop_event(c, timeout=10.0)
        assert body.get("reason") == "breakpoint", body
        assert body.get("line") == 6, \
            f"conditional bp must fire at line 6, not 5; got {body!r}"
        vars_ = _vars_by_name(c)
        assert vars_.get("x") == "10", \
            f"x should be 10 at line 6 hook, got vars={vars_!r}"
        c.request("continue")
        c.expect_no_event("stopped", window=0.4)
        c.wait_event("terminated", timeout=5.0)


def scn_log_point(matlabc, program):
    """Log point emits an interpolated 'output' event; no 'stopped'."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(
            c,
            breakpoints=[{"line": 6, "logMessage": "x={x} y={y}"}],
        )
        ev = c.wait_event(
            "output",
            timeout=5.0,
            predicate=lambda m: (m.get("body") or {}).get("category") == "console",
        )
        assert (ev.get("body") or {}).get("output") == "x=10 y=20\n", \
            f"unexpected log output: {ev!r}"
        c.expect_no_event("stopped", window=0.4)
        c.wait_event("terminated", timeout=5.0)


# --- entry point -------------------------------------------------------------

def all_scenarios():
    g = globals()
    return [(name, g[name]) for name in sorted(g) if name.startswith("scn_")]
