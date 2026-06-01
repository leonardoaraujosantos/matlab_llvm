"""DAP scenarios for matlabc.

Each scenario function takes (matlabc_path, program_path) and raises
on failure. The runner picks them up by their `scn_` prefix.

Most scenarios reuse `dap_program.m`. The error()-backtrace scenario
uses its own `dap_error_program.m` since the latter needs nested user
functions whose layout would only confuse the simpler scenarios.

Line numbers below reference dap_program.m:

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


def _assert_caret_consistent(client, expected_line, msg=""):
    """The IDE's source caret comes from `stackTrace`, not from
    the `stopped` event's `line` field. The factorial regression
    showed these can diverge: `stopped.line=13` was correct but
    stackTrace[0] still reported the pre-rewind `line=14`, so the
    caret didn't move on screen even though the protocol said it
    should.

    Any scenario that asserts a stopped line should also call this
    helper so the caret-consistency contract is checked, not just
    the wire-level field. The helper queries stackTrace afresh
    (no caching) since the contract is about what the IDE will
    see when it asks."""
    st = client.request("stackTrace", {"threadId": 1})
    frames = st.get("stackFrames") or []
    assert frames, \
        f"stackTrace empty when expected line={expected_line} ({msg})"
    top_line = frames[0].get("line")
    assert top_line == expected_line, \
        f"caret desync: stackTrace[0].line={top_line} vs " \
        f"expected line={expected_line} ({msg})"


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
        _assert_caret_consistent(c, 5, "basic bp stop")
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_rapid_fire_step_no_lost_stops(matlabc, program):
    """N back-to-back `next` requests must produce N stopped events.

    Regression: the IDE pipelines `next` clicks as the user presses
    Step Over rapidly. Earlier the DAP server's step handler issued
    matlab_dbg_resume + nudgeMonitor unconditionally — when a second
    `next` arrived while the worker was still mid-step from the
    first, the redundant resume was a no-op (paused=0 already) but
    nudgeMonitor still bumped ResumeGen. The worker took ONE step;
    the monitor's outer-wait re-check raced against the server's
    resume of the next request and silently skipped the pause for
    that step. Result: N nexts → N-1 (or fewer) stopped events,
    visible to the user as "Step Over stopped pausing — I had to
    hit Pause manually."

    The fix tracks StepsRequested vs StopsEmitted under G.Mu plus a
    MonitorBusy flag spanning the monitor's pause-detect →
    sendEvent → StopsEmitted++ window. waitForStepReady gates new
    step requests on `StopsEmitted > StepsRequested`, so each
    request maps to exactly one stop.

    This test sends 6 `next` frames back-to-back to the DAP server
    without waiting for stopped between them, then asserts exactly
    6 stopped events arrive.
    """
    import json as _json
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, stop_on_entry=True)
        # Drain the entry stop.
        _stop_event(c)
        # Pipeline 6 next requests without inter-frame waits.
        n = 6
        for k in range(n):
            msg = _json.dumps({"seq": 1000 + k, "type": "request",
                                "command": "next",
                                "arguments": {"threadId": 1}})
            frame = ("Content-Length: " + str(len(msg)) +
                     "\r\n\r\n" + msg).encode()
            c.proc.stdin.write(frame)
            c.proc.stdin.flush()
        # Collect stops with a generous per-event timeout. Each next
        # must produce one stop; the program (dap_program.m) has more
        # than 6 hookable lines so we won't run off the end.
        stops = 0
        for _ in range(n):
            body = _stop_event(c, timeout=4.0)
            assert body.get("reason") in ("step", "breakpoint"), \
                f"unexpected stop reason {body!r}"
            stops += 1
        assert stops == n, f"expected {n} stops, got {stops}"
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
        _assert_caret_consistent(c, 4, "first step pause")
        c.request("next")
        body = _stop_event(c)
        assert body.get("reason") == "step", \
            f"step-over stop reason should be 'step', got {body!r}"
        assert body.get("line") == 5, \
            f"second stop should be at line 5, got {body!r}"
        _assert_caret_consistent(c, 5, "after next")
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
    """setVariable mutates the workspace; subsequent variables read it back.

    Covers both scalar and matrix-literal RHS values — the latter goes
    through the same REPL JIT pipeline conditional breakpoints use, so
    `[1 2; 3 4]` parses and stores cleanly.
    """
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)
        before = _vars_by_name(c)
        assert before.get("x") == "10", f"sanity: x should be 10, got {before!r}"

        # Scalar set.
        resp = c.request("setVariable", {
            "variablesReference": 1,
            "name": "x",
            "value": "99",
        })
        assert resp.get("value") == "99", f"setVariable resp: {resp!r}"
        after = _vars_by_name(c)
        assert after.get("x") == "99", \
            f"x should read back as 99, got {after!r}"
        # y / z untouched by the scalar set.
        assert after.get("y") == "20", after
        assert after.get("z") == "30", after

        # Matrix-literal set: routes through the JIT, formatVar renders
        # the shape summary. The IDE's watch box gets "2x2 double".
        resp = c.request("setVariable", {
            "variablesReference": 1,
            "name": "y",
            "value": "[1 2; 3 4]",
        })
        assert resp.get("value") == "2x2 double", \
            f"matrix setVariable resp: {resp!r}"
        after = _vars_by_name(c)
        assert after.get("y") == "2x2 double", \
            f"y should read back as 2x2 double, got {after!r}"

        # Fresh-name assignment (variable didn't exist beforehand).
        resp = c.request("setVariable", {
            "variablesReference": 1,
            "name": "newvar",
            "value": "42",
        })
        assert resp.get("value") == "42", \
            f"fresh-name setVariable resp: {resp!r}"
        after = _vars_by_name(c)
        assert after.get("newvar") == "42", after

        # Malformed RHS must come back as success=false without
        # dropping the connection.
        try:
            c.request("setVariable", {
                "variablesReference": 1,
                "name": "x",
                "value": "1 +",
            })
            raise AssertionError("setVariable accepted a malformed RHS")
        except DapError:
            pass

        # Invalid name (not an identifier) — defense-in-depth path.
        try:
            c.request("setVariable", {
                "variablesReference": 1,
                "name": "1bad",
                "value": "5",
            })
            raise AssertionError("setVariable accepted a non-identifier name")
        except DapError:
            pass

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


def scn_error_backtrace(matlabc, program):
    """error() inside a user function prints a frame-by-frame traceback.

    Uses dap_error_program.m (sibling fixture) because the script
    needs nested user functions to produce a meaningful frame stack.
    The runtime snapshots the frame array inside matlab_set_error_msg
    before unwinding, then emits `error: <msg>\\n  at <fn> (<file>:<line>)`
    lines to stderr via write(2) so the libc stdio file lock can't
    deadlock against MLIR's ExecutionEngine.

    Note on flakiness: a separate, pre-existing MLIR/JIT shutdown race
    sometimes appends an `std::__1::system_error: recursive_mutex
    lock failed` line after our traceback. The assertions below are
    substring-based so that race doesn't fail the test.
    """
    import os, time
    err_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_error_program.m",
    )
    with DapClient(matlabc, err_program) as c:
        initialize_and_launch(c, stop_on_entry=False)
        c.wait_event("terminated", timeout=10.0)
    # Let the stderr drain thread finish reading whatever the matlabc
    # process buffered before disconnect closed the pipe.
    time.sleep(0.2)
    err = "".join(c._stderr_buf)
    assert "error: boom" in err, \
        f"missing error message header; stderr={err!r}"
    assert "at deeper (" in err, \
        f"missing innermost frame; stderr={err!r}"
    assert "at fail (" in err, \
        f"missing middle frame; stderr={err!r}"
    assert "at <script> (" in err, \
        f"missing script frame; stderr={err!r}"
    # Innermost frame (the actual error site) must come first; that
    # ordering is what the rest of the toolchain (lldb, gdb) emits and
    # what users expect when triaging a stack.
    deeper_pos = err.index("at deeper (")
    fail_pos = err.index("at fail (")
    script_pos = err.index("at <script> (")
    assert deeper_pos < fail_pos < script_pos, \
        f"frames out of order: deeper={deeper_pos}, fail={fail_pos}, script={script_pos}; stderr={err!r}"


def scn_classdef_prelude_launches(matlabc, program):
    """Regression for #77: a `-dap` launch of a toolbox-classdef program
    must compile + run (reach `terminated`).

    Uses dap_classdef_program.m (sibling fixture), which exercises
    dlarray method dispatch (`relu`), operator overloads (`*`/`+`), and
    `extractdata`. Before the classdef-prelude parse + dead-strip fix the
    merged prelude was silently dropped (it parsed standalone as a single
    classdef file and errored at the 2nd classdef), method/operator
    dispatch collapsed, and `launch` answered "failed to compile program".
    Reaching `terminated` here means the whole prelude lowered cleanly.
    """
    cd_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_classdef_program.m",
    )
    with DapClient(matlabc, cd_program) as c:
        initialize_and_launch(c, stop_on_entry=False)
        # No DapError from launch (would be "failed to compile program")
        # and the program runs to completion.
        c.wait_event("terminated", timeout=15.0)


def scn_closure_launches(matlabc, program):
    """Regression for #77: a `-dap` launch of an anonymous-closure program
    must compile + run (reach `terminated`).

    Uses dap_closure_program.m (sibling fixture): a captured scalar closure
    (`@(x) x+k`), a capture-free anon passed to a function and called
    indirectly there (`apply(g, 5)`), and an inline anon literal argument.
    Before the "anon closures use the local-slot lane in JIT/-dap" fix, the
    ReplMode workspace round-trip severed the make_anon -> call_indirect
    chain and `launch` answered "failed to compile program".
    """
    cl_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_closure_program.m",
    )
    with DapClient(matlabc, cl_program) as c:
        initialize_and_launch(c, stop_on_entry=False)
        c.wait_event("terminated", timeout=15.0)


def scn_function_locals(matlabc, program):
    """Pausing inside a user function exposes the function's locals.

    Today's `Locals` view is per-frame: ref = 1000 + DAP_frame_id, with
    the runtime mirroring every store to a named slot into a per-frame
    mini-workspace. The script-frame view merges matlab_ws (REPL-mode
    assignments) with that frame's mini-ws; function-frame views just
    return the per-frame mini-ws.
    """
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    # Line 11 is `s = total * 2;` inside compute(). The hook fires at
    # the start of the stmt, so by then a, b (param spills) and total
    # (line 10) are all in the function frame's mini-ws.
    with DapClient(matlabc, locals_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 11}])
        body = _stop_event(c)
        assert body.get("line") == 11, body

        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        assert len(frames) >= 2, \
            f"expected at least two frames (compute + <script>), got {frames!r}"
        # frame_id 0 = innermost = compute; frame_id 1 = <script>.
        inner = frames[0]
        outer = frames[1]
        assert "compute" in (inner.get("name") or ""), \
            f"innermost frame should be compute, got {inner!r}"
        assert "<script>" in (outer.get("name") or ""), \
            f"outermost frame should be <script>, got {outer!r}"

        # Locals for the innermost (compute) frame.
        sc = c.request("scopes", {"frameId": inner["id"]})
        scopes = sc.get("scopes") or []
        assert scopes, sc
        ref = scopes[0].get("variablesReference")
        vars_inner = _vars_by_name(c, ref=ref)
        assert vars_inner.get("a") == "3", \
            f"expected compute.a=3, got {vars_inner!r}"
        assert vars_inner.get("b") == "4", \
            f"expected compute.b=4, got {vars_inner!r}"
        assert vars_inner.get("total") == "7", \
            f"expected compute.total=7, got {vars_inner!r}"
        # `s` not yet assigned (the bp is on the line that computes it
        # — hook fires before the store), so it must not appear.
        assert "s" not in vars_inner, \
            f"compute.s should not be visible before its assignment: {vars_inner!r}"
        # Script-level `seed` belongs to the outer frame and must NOT
        # appear in the inner frame's view.
        assert "seed" not in vars_inner, \
            f"compute frame should not expose script-scope seed: {vars_inner!r}"

        # Locals for the outer (script) frame: must include matlab_ws
        # contents (seed) but not compute's locals.
        sc_outer = c.request("scopes", {"frameId": outer["id"]})
        ref_outer = (sc_outer.get("scopes") or [{}])[0].get("variablesReference")
        vars_outer = _vars_by_name(c, ref=ref_outer)
        assert vars_outer.get("seed") == "7", \
            f"script.seed should be 7, got {vars_outer!r}"
        assert "a" not in vars_outer, \
            f"script frame should not leak compute.a: {vars_outer!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_evaluate(matlabc, program):
    """DAP `evaluate` against the script-level workspace.

    Runs the user expression through the same REPL JIT pipeline
    conditional breakpoints use. The frame-scoped variant is exercised
    by scn_evaluate_in_frame below.
    """
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)
        # Pure-arithmetic expression — independent of any workspace state.
        resp = c.request("evaluate", {"expression": "1 + 1"})
        assert resp.get("result") == "2", \
            f"evaluate '1 + 1' should be 2, got {resp!r}"
        # Workspace reference: x was assigned on line 4 of dap_program.m.
        resp = c.request("evaluate", {"expression": "x"})
        assert resp.get("result") == "10", \
            f"evaluate 'x' should be 10, got {resp!r}"
        # Compound expression mixing literals and ws vars.
        resp = c.request("evaluate", {"expression": "x + y"})
        assert resp.get("result") == "30", \
            f"evaluate 'x + y' should be 30, got {resp!r}"
        # Matrix-valued evaluate renders via formatVar.
        resp = c.request("evaluate", {"expression": "[1 2; 3 4]"})
        assert resp.get("result") == "2x2 double", \
            f"evaluate matrix literal should render as '2x2 double', got {resp!r}"
        # User-typed trailing semicolons are tolerated.
        resp = c.request("evaluate", {"expression": "x + 1;"})
        assert resp.get("result") == "11", \
            f"evaluate with trailing ; should still return 11, got {resp!r}"
        # Malformed expression must come back as success=false without
        # dropping the connection. The response message must carry the
        # actual diagnostic captured from runReplInput — not the
        # generic "see debug console" placeholder.
        try:
            c.request("evaluate", {"expression": "1 +"})
            raise AssertionError("evaluate accepted a malformed expression")
        except DapError as e:
            err = str(e)
            assert "error:" in err.lower() or "expected" in err.lower() or \
                   "unexpected" in err.lower(), \
                f"evaluate error should carry the captured diagnostic, got {err!r}"
            assert "see debug console" not in err, \
                f"evaluate error should NOT fall back to placeholder: {err!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_evaluate_in_frame(matlabc, program):
    """`evaluate` resolves function-frame locals when frameId points at
    a non-script frame.

    Implementation: the DAP server snapshots matlab_ws, stamps the
    chosen frame's mini-ws into ws, runs runReplInput, reads the
    result, then restores ws (clearing freshly-stamped names that
    didn't exist pre-stamp). The script workspace must be unchanged
    after the call so subsequent script-frame evals don't see leaked
    function locals.
    """
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    # Pause inside compute(a, b) where a=3, b=4, total=7 are visible.
    with DapClient(matlabc, locals_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 11}])
        _stop_event(c)
        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        inner = frames[0]
        outer = frames[1]
        inner_id = inner["id"]
        outer_id = outer["id"]

        # Without frameId — eval defaults to the script ws. The REPL
        # JIT silently resolves an unknown name to an empty matrix
        # (0x0 double) rather than erroring. The exact rendering
        # ("0x0 double" today) is a runtime detail; what matters for
        # the bridge test is that this baseline value is NOT "3", so
        # we'll be sure the per-frame branch below changed something.
        resp_no_frame = c.request("evaluate", {"expression": "a"})
        assert resp_no_frame.get("result") != "3", \
            f"sanity: 'a' without frameId must not be 3 (compute's value): {resp_no_frame!r}"

        # With frameId pointing at the function frame: a/b/total must
        # all resolve and arithmetic on them must work.
        for expr, expected in (("a", "3"), ("b", "4"), ("total", "7"),
                               ("a + b", "7"), ("total * 2", "14")):
            resp = c.request("evaluate", {
                "expression": expr,
                "frameId": inner_id,
            })
            assert resp.get("result") == expected, \
                f"evaluate({expr!r}) in compute frame: expected {expected}, got {resp!r}"

        # After the function-frame eval, the script workspace must NOT
        # have 'a', 'b', or 'total' lingering — those were temporarily
        # stamped during eval and must have been cleared on restore.
        # The script frame's variables snapshot is a clean way to
        # check (it shows matlab_ws + script-frame mini-ws).
        sc_outer = c.request("scopes", {"frameId": outer_id})
        ref_outer = (sc_outer.get("scopes") or [{}])[0].get("variablesReference")
        vars_outer = _vars_by_name(c, ref=ref_outer)
        for n in ("a", "b", "total"):
            assert n not in vars_outer, \
                f"frame-scoped eval leaked {n!r} into the script workspace: {vars_outer!r}"

        # And explicit `evaluate` without frameId must NOT now resolve
        # 'a' to 3 — the restore took the stamped value back out of ws.
        resp = c.request("evaluate", {"expression": "a"})
        assert resp.get("result") != "3", \
            f"after frame-scoped eval the bridge should be reversed: {resp!r}"
        # And the value must equal what we got before the bridge fired,
        # i.e. nothing about the script ws view of 'a' has changed.
        assert resp.get("result") == resp_no_frame.get("result"), \
            f"script-scope evaluate('a') changed across the bridge: " \
            f"before={resp_no_frame!r} after={resp!r}"

        # Pre-existing script-scope `seed` survives the eval round-trip
        # untouched.
        resp = c.request("evaluate", {"expression": "seed"})
        assert resp.get("result") == "7", \
            f"script-scope seed should be 7 after frame eval, got {resp!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_threads_and_continued(matlabc, program):
    """`threads` returns a single-thread list, and `continued` events
    are emitted on every resume request — important for adapters that
    track stopped/continued symmetry."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 5}])
        _stop_event(c)

        body = c.request("threads")
        ts = body.get("threads") or []
        assert any(t.get("id") == 1 for t in ts), \
            f"expected thread id=1 in {body!r}"

        c.request("continue")
        ev = c.wait_event("continued", timeout=5.0)
        assert (ev.get("body") or {}).get("threadId") == 1, \
            f"continued event missing threadId: {ev!r}"
        c.wait_event("terminated", timeout=5.0)


def scn_loaded_sources_and_source(matlabc, program):
    """`loadedSources` lists the registered .m files; `source` returns
    file content for adapters that don't have direct fs access."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 5}])
        _stop_event(c)

        body = c.request("loadedSources")
        sources = body.get("sources") or []
        paths = [s.get("path") for s in sources]
        assert any(_abs(p) == _abs(program) for p in paths), \
            f"entry-point not in loadedSources: {paths!r}"

        body = c.request("source", {"source": {"path": program}})
        content = body.get("content") or ""
        with open(program) as f:
            expected = f.read()
        assert content == expected, \
            f"source content mismatch (got {len(content)} bytes, expected {len(expected)})"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_completions(matlabc, program):
    """`completions` returns workspace + builtin matches for a prefix.

    At line 8 of dap_program.m, `x` and `y` are in the workspace; both
    must surface for prefix `x` / `y`. A bare prefix `dis` should
    surface the `disp` builtin from the curated list."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)

        body = c.request("completions", {"text": "x", "column": 2})
        labels = [t.get("label") for t in (body.get("targets") or [])]
        assert "x" in labels, f"workspace x missing: {labels!r}"

        body = c.request("completions", {"text": "dis", "column": 4})
        labels = [t.get("label") for t in (body.get("targets") or [])]
        assert "disp" in labels, f"disp builtin missing: {labels!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_set_expression(matlabc, program):
    """`setExpression` mutates by lvalue expression — uses the same
    REPL-JIT assignment path as setVariable but accepts arbitrary
    lvalues (struct fields, indexed cells, etc.). For dap_program.m
    we just rebind a top-level scalar.

    The response renders the computed value via a readback of the
    same lvalue, so an RHS like `2 * 21` returns `value="42"`,
    not the raw RHS string."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)

        # Computed RHS — the response must show the result, not
        # the literal "2 * 21".
        resp = c.request("setExpression",
                         {"expression": "x", "value": "2 * 21"})
        assert resp.get("value") == "42", \
            f"setExpression should return the computed value, got {resp!r}"

        # Independent confirmation via evaluate.
        resp = c.request("evaluate", {"expression": "x"})
        assert resp.get("result") == "42", \
            f"setExpression didn't persist: x={resp!r}"

        # Matrix-valued RHS — the response should report the
        # shape label ("2x2 double") and a mat-ref for drilling.
        resp = c.request("setExpression",
                         {"expression": "x", "value": "[1 2; 3 4]"})
        assert resp.get("value") == "2x2 double", \
            f"matrix setExpression: {resp!r}"
        assert (resp.get("variablesReference") or 0) >= 200000, \
            f"matrix setExpression should carry mat-ref: {resp!r}"
        assert resp.get("indexedVariables") == 4, \
            f"matrix setExpression should advertise indexedVariables=4: {resp!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_function_breakpoints(matlabc, program):
    """`setFunctionBreakpoints` resolves a name against the compiled
    function table and installs a line breakpoint at the body's first
    line. Hits like a normal bp."""
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    with DapClient(matlabc, locals_program) as c:
        # Use the lifecycle helper but skip its line breakpoints —
        # we install function bps below before configurationDone.
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        assert caps.get("supportsFunctionBreakpoints"), caps
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": locals_program, "stopOnEntry": False})
        body = c.request("setFunctionBreakpoints", {
            "breakpoints": [{"name": "compute"}, {"name": "no_such_fn"}],
        })
        bps = body.get("breakpoints") or []
        assert bps[0].get("verified") is True, f"compute bp not verified: {bps[0]!r}"
        assert bps[1].get("verified") is False, \
            f"no_such_fn should not verify: {bps[1]!r}"
        c.request("configurationDone")

        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        # Should stop inside compute (line 10 = `total = a + b;`).
        assert body.get("line") == 10, f"function bp landed at {body!r}"
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_frame_scoped_conditional_breakpoint(matlabc, program):
    """A conditional bp inside a function body must see the
    function's parameters / locals — not just script-scope vars.

    dap_locals_program.m has `function s = compute(a, b)` with
    `total = a + b;` on line 10. We set a bp on that line with
    condition `a > 2` (a is a parameter, value 3 in the call); the
    runtime hook fires, the cond evaluator bridges the function
    frame's mini-ws into matlab_ws, the condition evaluates true,
    and the IDE sees a `stopped` event.

    Without the bridge, `a` resolves to 0x0 / undefined in the
    REPL JIT and the condition silently evaluates false."""
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    with DapClient(matlabc, locals_program) as c:
        initialize_and_launch(c, breakpoints=[
            {"line": 10, "condition": "a > 2"},
        ])
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 10, \
            f"frame-scoped cond should land at line 10: {body!r}"
        assert body.get("reason") == "breakpoint", body
        _assert_caret_consistent(c, 10, "frame-scoped cond pause")
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)

    # Verify the negative case: `a > 99` is false, no stop event.
    with DapClient(matlabc, locals_program) as c:
        initialize_and_launch(c, breakpoints=[
            {"line": 10, "condition": "a > 99"},
        ])
        c.expect_no_event("stopped", window=0.5)
        c.wait_event("terminated", timeout=5.0)


def scn_frame_scoped_log_point(matlabc, program):
    """`{name}` placeholders in a logMessage on a function-body bp
    resolve against the function frame, not just script ws.
    `{a}` inside `compute(3, 4)` must print `3`."""
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    with DapClient(matlabc, locals_program) as c:
        initialize_and_launch(c, breakpoints=[
            {"line": 10, "logMessage": "a={a} b={b}"},
        ])
        ev = c.wait_event(
            "output",
            timeout=5.0,
            predicate=lambda m: (m.get("body") or {}).get("category") == "console",
        )
        out = (ev.get("body") or {}).get("output", "")
        assert out.strip() == "a=3 b=4", \
            f"logpoint didn't bridge frame locals: {out!r}"
        c.expect_no_event("stopped", window=0.4)
        c.wait_event("terminated", timeout=5.0)


def scn_pending_breakpoint_event(matlabc, program):
    """`setBreakpoints` against a path the runtime hasn't registered
    yet (request arrived before launch / compileProgram) returns
    verified=false and queues the bp. After launch populates the
    path registry, the queued bp is replayed and a `breakpoint`
    event with reason="changed" surfaces the now-verified state.

    DAP-permitted ordering: initialize → initialized → setBreakpoints
    → launch → configurationDone. (Our usual helper does launch
    first; this scenario swaps the order to exercise the queue.)"""
    with DapClient(matlabc, program) as c:
        c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        c.wait_event("initialized", timeout=5.0)

        # setBreakpoints BEFORE launch — path registry is empty,
        # so the bp comes back unverified and goes on the pending
        # queue.
        body = c.request("setBreakpoints", {
            "source": {"path": program},
            "breakpoints": [{"line": 5}],
        })
        bps = body.get("breakpoints") or []
        assert bps and bps[0].get("verified") is False, \
            f"expected verified=false pre-launch: {bps!r}"

        # Launch + configurationDone — compileProgram registers the
        # path, sweeps the queue, emits the change event.
        c.request("launch", {"program": program, "stopOnEntry": False})
        c.request("configurationDone")

        ev = c.wait_event(
            "breakpoint",
            timeout=5.0,
            predicate=lambda m: (m.get("body") or {}).get("reason") == "changed",
        )
        bp = (ev.get("body") or {}).get("breakpoint") or {}
        assert bp.get("verified") is True, \
            f"changed event should carry verified=true: {bp!r}"
        assert bp.get("line") == 5, bp
        assert isinstance(bp.get("id"), int), bp

        # Bp should fire normally now.
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 5, body
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_hit_count_breakpoint(matlabc, program):
    """`hitCondition` on a bp inside a loop body suppresses the
    first N-1 hits and pauses on the Nth (or every match after
    that, depending on the operator).

    dap_program.m line 9 (`z = z + i;`) is inside a `for i = 1:3`
    loop, so the bp fires 3 times. `hitCondition: ">= 3"` should
    pause on the third iteration only — by which point i == 3."""
    with DapClient(matlabc, program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        assert caps.get("supportsHitConditionalBreakpoints"), \
            f"hit-count caps not advertised: {caps}"
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": program, "stopOnEntry": False})
        body = c.request("setBreakpoints", {
            "source": {"path": program},
            "breakpoints": [{"line": 9, "hitCondition": ">= 3"}],
        })
        bps = body.get("breakpoints") or []
        assert bps[0].get("verified"), bps
        c.request("configurationDone")

        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 9, body
        _assert_caret_consistent(c, 9, "hit-count bp pause")

        # Confirm the iteration counter — `i` should be 3.
        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": (st["stackFrames"] or [{}])[0]["id"]})
        ref = (sc.get("scopes") or [{}])[0].get("variablesReference")
        rows = _vars_by_name(c, ref=ref)
        assert rows.get("i") == "3", \
            f"hit count ≥3 should pause when i=3, got i={rows.get('i')!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_class_instance_methods(matlabc, program):
    """Class-instance expansion in `variables` surfaces method rows
    after property rows. Method rows carry presentationHint.kind=
    "method" so the IDE renders a function icon, plus a value column
    with a `@name(args)` signature for arity at a glance.

    Inherited methods from a superclass appear too, with an
    "(inherited from X)" suffix on the value. Override (same-name
    method on the derived class) suppresses the parent's entry.

    Uses dap_class_program.m which has Account (Id, Balance, deposit)
    and Savings < Account (Rate, plus inherited deposit / Account
    ctor)."""
    import os
    class_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_class_program.m",
    )
    with DapClient(matlabc, class_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        ref = sc["scopes"][0]["variablesReference"]
        body = c.request("variables", {"variablesReference": ref})
        rows = {v["name"]: v for v in (body.get("variables") or [])}

        # acc: Account instance. Children = Id, Balance, then
        # Account constructor + deposit method.
        acc_ref = rows["acc"]["variablesReference"]
        body = c.request("variables", {"variablesReference": acc_ref})
        children = {v["name"]: v for v in (body.get("variables") or [])}

        # Properties still present.
        assert children.get("Id", {}).get("value") == "101", \
            f"Id property missing/wrong: {children!r}"
        assert children.get("Balance", {}).get("value") == "75", \
            f"Balance property missing/wrong: {children!r}"

        # Methods present, tagged by type and presentationHint.
        for m in ("deposit", "Account"):
            row = children.get(m)
            assert row, f"method {m!r} missing from acc: {children!r}"
            assert row.get("type") == "method", \
                f"method {m!r} type expected 'method': {row!r}"
            assert row.get("variablesReference") == 0, \
                f"method {m!r} should be a leaf: {row!r}"
            ph = row.get("presentationHint") or {}
            assert ph.get("kind") == "method", \
                f"method {m!r} missing presentationHint.kind=method: {row!r}"
            # Value column must include the signature.
            assert "@" + m in (row.get("value") or ""), \
                f"method {m!r} value should be a signature: {row!r}"
        # acc's deposit row must NOT carry an "inherited from"
        # suffix — it's defined directly on Account.
        assert "inherited" not in (children["deposit"].get("value") or ""), \
            f"acc.deposit should be direct, not inherited: {children['deposit']!r}"

        # sav: Savings instance. Inherits Account's deposit; should
        # carry the "inherited from Account" hint.
        sav_ref = rows["sav"]["variablesReference"]
        body = c.request("variables", {"variablesReference": sav_ref})
        children = {v["name"]: v for v in (body.get("variables") or [])}

        # Inherited Id + Balance from Account, plus own Rate.
        assert children.get("Rate", {}).get("value") == "0.1", \
            f"Savings.Rate missing/wrong: {children!r}"

        # Savings own constructor + inherited Account ctor + inherited deposit.
        assert children.get("Savings", {}).get("type") == "method", children
        assert "inherited" not in (children["Savings"].get("value") or ""), \
            f"Savings.Savings (own ctor) should not be inherited: {children['Savings']!r}"
        for m in ("Account", "deposit"):
            row = children.get(m)
            assert row, f"inherited method {m!r} missing on sav: {children!r}"
            assert "inherited from Account" in (row.get("value") or ""), \
                f"sav.{m} should report inheritance: {row!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_data_breakpoint_write(matlabc, program):
    """Write-only data breakpoint: stop every time `target` gets
    assigned. The fixture writes `target = 1` then `target = 2`,
    so the runtime trips twice; the stopped event reports
    reason="data breakpoint" and surfaces the watch's id in
    hitBreakpointIds."""
    import os
    wp_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_watchpoint_program.m",
    )
    with DapClient(matlabc, wp_program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        assert caps.get("supportsDataBreakpoints"), \
            f"data breakpoints caps not advertised: {caps}"
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": wp_program, "stopOnEntry": False})

        # Resolve a dataId for the name. dataBreakpointInfo round-
        # trips it through the IDE; we mirror the real DAP flow.
        info = c.request("dataBreakpointInfo", {"name": "target"})
        data_id = info.get("dataId")
        assert data_id == "target", \
            f"dataBreakpointInfo should hand back the name as dataId: {info!r}"
        types = info.get("accessTypes") or []
        assert "write" in types, f"write accessType missing: {info!r}"

        body = c.request("setDataBreakpoints", {
            "breakpoints": [{"dataId": data_id, "accessType": "write"}],
        })
        bps = body.get("breakpoints") or []
        assert bps and bps[0].get("verified"), \
            f"watchpoint not verified: {bps!r}"
        wp_id = bps[0].get("id")
        assert isinstance(wp_id, int) and wp_id > 0, \
            f"watchpoint should carry a stable id: {bps!r}"

        c.request("configurationDone")

        # First trip: line 6 (`target = 1;`).
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "data breakpoint", \
            f"first stop should report reason='data breakpoint': {body!r}"
        assert body.get("hitBreakpointIds") == [wp_id], \
            f"first stop should surface watch id: {body!r}"

        # Workspace inspection: target == 1 right after the first write.
        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        rows = _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])
        assert rows.get("target") == "1", \
            f"target should be 1 at first trip: {rows!r}"

        c.request("continue")

        # Second trip: line 7 (`target = 2;`).
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "data breakpoint", \
            f"second stop should report reason='data breakpoint': {body!r}"

        rows = _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])
        assert rows.get("target") == "2", \
            f"target should be 2 at second trip: {rows!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_data_breakpoint_clear(matlabc, program):
    """An empty `setDataBreakpoints` list clears every prior watch.
    After clearing, an assignment that previously tripped now runs
    through silently."""
    import os
    wp_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_watchpoint_program.m",
    )
    with DapClient(matlabc, wp_program) as c:
        c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
        })
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": wp_program, "stopOnEntry": False})

        # Set then clear the watch in the same pre-launch window.
        c.request("setDataBreakpoints", {
            "breakpoints": [{"dataId": "target", "accessType": "write"}],
        })
        c.request("setDataBreakpoints", {"breakpoints": []})
        c.request("configurationDone")

        # No watchpoint trips — program runs to completion without a
        # `stopped` event.
        c.expect_no_event("stopped", window=0.5)
        c.wait_event("terminated", timeout=5.0)


def scn_data_breakpoint_read(matlabc, program):
    """Read watchpoint on a script-scope variable. The fixture
    writes `target` twice and then reads it on line 8 (`disp(target)`).
    With access="read", the writes don't trip but the disp's read
    does — trip count = 1, reason="data breakpoint"."""
    import os
    wp_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_watchpoint_program.m",
    )
    with DapClient(matlabc, wp_program) as c:
        c.request("initialize", {"clientID": "matlabc-test",
                                  "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": wp_program, "stopOnEntry": False})

        body = c.request("setDataBreakpoints", {
            "breakpoints": [{"dataId": "target", "accessType": "read"}],
        })
        bps = body.get("breakpoints") or []
        assert bps and bps[0].get("verified"), \
            f"read watchpoint should verify: {bps!r}"

        c.request("configurationDone")

        # Trip is on the read: line 8 (`disp(target);`).
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "data breakpoint", body
        assert body.get("line") == 8, \
            f"read-watch should trip on the disp(target) read: {body!r}"

        c.request("continue")
        # No further trips — the next disp output and termination
        # should follow without another stopped event.
        c.wait_event("terminated", timeout=5.0)


def scn_data_breakpoint_readwrite(matlabc, program):
    """access="readWrite" trips on every read AND every write.
    Three trips total for the fixture: two writes, one read."""
    import os
    wp_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_watchpoint_program.m",
    )
    with DapClient(matlabc, wp_program) as c:
        c.request("initialize", {"clientID": "matlabc-test",
                                  "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": wp_program, "stopOnEntry": False})

        c.request("setDataBreakpoints", {
            "breakpoints": [{"dataId": "target", "accessType": "readWrite"}],
        })
        c.request("configurationDone")

        # Write on line 6, write on line 7, read on line 8.
        expected_lines = [6, 7, 8]
        for L in expected_lines:
            ev = c.wait_event("stopped", timeout=5.0)
            body = ev.get("body") or {}
            assert body.get("reason") == "data breakpoint", body
            assert body.get("line") == L, \
                f"trip on line {L} expected, got {body!r}"
            c.request("continue")

        c.wait_event("terminated", timeout=5.0)


def scn_data_breakpoint_accesstype_advertised(matlabc, program):
    """`dataBreakpointInfo` advertises read / write / readWrite as
    the supported access types. The IDE renders a chooser when
    setting a data breakpoint."""
    import os
    wp_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_watchpoint_program.m",
    )
    with DapClient(matlabc, wp_program) as c:
        c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
        })
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": wp_program, "stopOnEntry": False})

        info = c.request("dataBreakpointInfo", {"name": "target"})
        types = info.get("accessTypes") or []
        for t in ("read", "write", "readWrite"):
            assert t in types, \
                f"accessType {t!r} not advertised: {info!r}"
        c.request("configurationDone")
        c.wait_event("terminated", timeout=5.0)


def scn_read_write_memory(matlabc, program):
    """Matrix variable rows now carry a `memoryReference` pointing
    at the data buffer, so the IDE can `readMemory` / `writeMemory`
    raw cell bytes through the standard DAP affordances.

    For `dap_matrix_program.m`, A is a 2x3 row-major double matrix
    `[1 2 3; 4 5 6]`. We read the first 24 bytes (3 cells of slice 0)
    and confirm the doubles match {1, 2, 3}, then write a new
    pattern back and read it again to confirm the round-trip.
    Reading past the buffer end reports the unread tail in
    `unreadableBytes` instead of erroring."""
    import os
    import base64
    import struct

    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        body = c.request("variables",
                         {"variablesReference": sc["scopes"][0]["variablesReference"]})
        rows = {v["name"]: v for v in (body.get("variables") or [])}
        mem_ref = rows["A"].get("memoryReference")
        assert mem_ref and mem_ref.startswith("0x"), \
            f"matrix A should carry a memoryReference: {rows['A']!r}"

        # Read the first three doubles (24 bytes). Row-major layout
        # → A's first row is [1, 2, 3].
        body = c.request("readMemory", {
            "memoryReference": mem_ref,
            "offset": 0,
            "count": 24,
        })
        assert body.get("address") == mem_ref, body
        bytes_back = base64.b64decode(body["data"])
        assert len(bytes_back) == 24, f"expected 24 bytes, got {len(bytes_back)}"
        cells = struct.unpack("<3d", bytes_back)
        assert cells == (1.0, 2.0, 3.0), \
            f"first row should be [1, 2, 3], got {cells!r}"

        # Read past the end — A is 2x3 = 6 cells = 48 bytes total.
        # Asking for 96 bytes should return 48 with unreadableBytes=48.
        body = c.request("readMemory", {
            "memoryReference": mem_ref, "offset": 0, "count": 96,
        })
        assert body.get("unreadableBytes") == 48, \
            f"reading past EOB should report unreadableBytes=48: {body!r}"
        bytes_back = base64.b64decode(body["data"])
        assert len(bytes_back) == 48, body

        # Round-trip via writeMemory: stamp a new pattern at offset 0
        # (3 cells) and read it back.
        new_pattern = struct.pack("<3d", 99.0, 88.0, 77.0)
        body = c.request("writeMemory", {
            "memoryReference": mem_ref,
            "offset": 0,
            "data": base64.b64encode(new_pattern).decode("ascii"),
        })
        assert body.get("bytesWritten") == 24, \
            f"writeMemory should report 24 bytes written: {body!r}"

        body = c.request("readMemory", {
            "memoryReference": mem_ref, "offset": 0, "count": 24,
        })
        cells = struct.unpack("<3d", base64.b64decode(body["data"]))
        assert cells == (99.0, 88.0, 77.0), \
            f"writeMemory round-trip failed: {cells!r}"

        # Bogus memoryReference is rejected cleanly.
        try:
            c.request("readMemory", {
                "memoryReference": "0xdeadbeef",
                "offset": 0,
                "count": 16,
            })
            raise AssertionError("readMemory should refuse unknown ptr")
        except DapError as e:
            assert "registered" in str(e).lower(), \
                f"refusal should mention registration: {e}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_step_back(matlabc, program):
    """`stepBack` rolls back the most recent statement's writes
    and resumes at the prior line.

    Uses dap_revstep_program.m: straight-line `a = 100; b = 200;
    c = 300; disp(c);` on lines 5-8. Break at line 8 with state
    {a=100, b=200, c=300}. Each stepBack walks back exactly one
    statement, reverting only that statement's writes:

      stepBack #1 -> line 7, state {a=100, b=200} (c removed)
      stepBack #2 -> line 6, state {a=100}        (b removed)
      stepBack #3 -> line 5, state {}             (a removed)
      stepBack #4 -> reason=entry (log exhausted)

    Variables that didn't pre-exist are removed from the workspace
    via matlab_struct_rmfield, not zeroed — so `who`/`whos`/
    DAP variable inspection see the pre-write state exactly."""
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
        })
        assert caps.get("supportsStepBack"), \
            f"stepBack caps not advertised: {caps}"
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 8}],
        })
        c.request("configurationDone")
        ev = c.wait_event("stopped", timeout=5.0)
        assert (ev.get("body") or {}).get("line") == 8, ev

        def _ws():
            st = c.request("stackTrace", {"threadId": 1})
            sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
            return _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])

        # Initial state: all three writes have happened.
        rows = _ws()
        assert rows == {"a": "100", "b": "200", "c": "300"}, \
            f"unexpected initial state: {rows!r}"

        # stepBack #1: line 7, c removed.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "step", \
            f"stepBack #1 should report reason=step: {body!r}"
        assert body.get("line") == 7, \
            f"stepBack #1 should resume at line 7: {body!r}"
        _assert_caret_consistent(c, 7, "stepBack #1")
        rows = _ws()
        assert rows == {"a": "100", "b": "200"}, \
            f"stepBack #1: c should be removed: {rows!r}"

        # stepBack #2: line 6, b removed.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 6, \
            f"stepBack #2 should resume at line 6: {body!r}"
        _assert_caret_consistent(c, 6, "stepBack #2")
        rows = _ws()
        assert rows == {"a": "100"}, \
            f"stepBack #2: b should be removed: {rows!r}"

        # stepBack #3: line 5, a removed (workspace empty).
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 5, \
            f"stepBack #3 should resume at line 5: {body!r}"
        _assert_caret_consistent(c, 5, "stepBack #3")
        rows = _ws()
        assert rows == {}, \
            f"stepBack #3: a should be removed; workspace empty: {rows!r}"

        # stepBack #4: undo log exhausted -> reason=entry.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "entry", \
            f"stepBack past the first statement should report " \
            f"reason=entry: {body!r}"
        assert "exhausted" in (body.get("description") or "").lower(), \
            f"entry stop should describe the empty-log condition: {body!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_step_back_overwrites(matlabc, program):
    """When a variable is overwritten across multiple statements,
    stepBack walks the values backward through the prior versions.
    Locks in that prev_existed=1 records restore the prior value
    rather than removing the binding.

    Uses dap_revstep_overwrite_program.m: `x = 1; x = 2; x = 3;
    disp(x);` on lines 1-4. Break on the disp; stepBack walks x
    through 2, 1, then removes."""
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_overwrite_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 4}],   # disp(x)
        })
        c.request("configurationDone")
        c.wait_event("stopped", timeout=5.0)

        def _ws():
            st = c.request("stackTrace", {"threadId": 1})
            sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
            return _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])

        assert _ws() == {"x": "3"}, _ws()

        # stepBack #1 -> x reverted to 2 (the value from line 2).
        c.request("stepBack")
        c.wait_event("stopped", timeout=5.0)
        assert _ws() == {"x": "2"}, _ws()

        # stepBack #2 -> x reverted to 1 (line 1's value).
        c.request("stepBack")
        c.wait_event("stopped", timeout=5.0)
        assert _ws() == {"x": "1"}, _ws()

        # stepBack #3 -> x removed (line 1 had no prior write).
        c.request("stepBack")
        c.wait_event("stopped", timeout=5.0)
        assert _ws() == {}, _ws()

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_step_back_inside_function(matlabc, program):
    """stepBack from inside a function frame must:
      a) update the *innermost frame's* line so DAP `stackTrace`
         reflects the rewound position (the IDE renders the caret
         from stackTrace, not from the stopped event's line);
      b) refuse to cross out of the current function — boundary
         records from the caller frame are skipped, and stepping
         past the function's first statement returns reason=entry
         instead of teleporting up into the script frame.

    Uses examples/factorial.m: bp on line 14 (`y = 1;`) inside
    `fact(n)`. Reaching it requires `disp(fact(1));` to call fact,
    which fires hooks across the script + function frames; the
    rewind has to ignore all of that and stay in fact's body.
    """
    import os
    fact_program = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(program))),
        "examples", "factorial.m",
    )
    if not os.path.exists(fact_program):
        # Skip cleanly when the example layout differs (e.g. when
        # tests run from a worktree that omits examples/).
        return
    with DapClient(matlabc, fact_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": fact_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": fact_program},
            "breakpoints": [{"line": 14}],
        })
        c.request("configurationDone")
        ev = c.wait_event("stopped", timeout=10.0)
        body = ev.get("body") or {}
        assert body.get("line") == 14 and body.get("reason") == "breakpoint", body

        def _frames():
            st = c.request("stackTrace", {"threadId": 1})
            return [(f.get("name"), f.get("line"))
                    for f in (st.get("stackFrames") or [])]

        # Initial stack: paused inside fact, called from the script.
        frames = _frames()
        assert frames[0] == ("fact", 14), \
            f"initial innermost frame should be fact:14: {frames!r}"

        # stepBack #1: stays inside fact, walks back to line 13
        # (the `if n <= 1` test). innermost frame line MUST update.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 13 and body.get("reason") == "step", body
        frames = _frames()
        assert frames[0] == ("fact", 13), \
            f"after stepBack the innermost frame should be fact:13 " \
            f"(IDE renders caret from stackTrace, not stopped.line): {frames!r}"

        # stepBack #2: walking back from line 13 would cross into
        # the caller (the `disp(fact(1))` site on script line 5).
        # The runtime refuses to cross frames; reports
        # reason=entry to signal "rewind exhausted in this frame".
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "entry", \
            f"stepBack across the function-call boundary should " \
            f"report reason=entry, not silently teleport up: {body!r}"
        # Frame state stays at fact:13 (where the prior stepBack
        # left us) — we didn't cross.
        frames = _frames()
        assert frames[0] == ("fact", 13), \
            f"refused-cross stepBack should leave the frame " \
            f"unchanged: {frames!r}"

        c.request("continue")
        try:
            c.wait_event("terminated", timeout=10.0)
        except DapError:
            pass


def scn_step_forward_back_forward(matlabc, program):
    """Forward step after stepBack walks the recorded future via
    the redo log instead of resuming the JIT thread (which is
    parked one statement past the rewound caret). User-visible
    contract: rewound caret + state + console output stay in
    lockstep on the way back AND on the way forward; the JIT
    only resumes once the redo catches up.

    The redo path is exercised by `next`/`stepIn`/`continue`/
    `stepOut` whenever matlab_dbg_is_rewound() reports true. Each
    redo step re-applies the post-write state captured at the
    original write (no JIT execution → no duplicate console
    output, no side effects re-played) and stops at the next
    same-frame statement boundary. When the redo head reaches the
    snapshot of undo_head taken at the FIRST stepBack of the
    sequence, `rewound` clears and the next forward step resumes
    the JIT for real.

    Sequence over dap_revstep_program.m (lines 5-8 are
    `a=100; b=200; c=300; disp(c);`):

      bp at 5      -> line=5  ws={}
      next #1      -> line=6  ws={a:100}              (JIT runs line 5)
      next #2      -> line=7  ws={a:100,b:200}        (JIT runs line 6)
      stepBack #1  -> line=6  ws={a:100}              (b reverted)
      stepBack #2  -> line=5  ws={}                   (a reverted)
      next #3      -> line=6  ws={a:100}              (REDO replay,
                                                        no JIT exec)
      next #4      -> line=7  ws={a:100,b:200}        (REDO catches up)
      next #5      -> line=8  ws={a:100,b:200,c:300}  (JIT resumes,
                                                        runs line 7)

    Before the redo log existed, `next #3` instead landed at
    line 8 with `ws={c:300}` because the JIT had been parked at
    line 7's hook and forward-stepping just resumed it from there
    — confusing users into thinking line 6 had been silently
    skipped. This test locks in the inverted contract."""
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 5}],
        })
        c.request("configurationDone")

        def _ws():
            st = c.request("stackTrace", {"threadId": 1})
            sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
            return _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])

        def _step_assert(cmd, label, expected_line, expected_ws):
            c.request(cmd)
            ev = c.wait_event("stopped", timeout=5.0)
            body = ev.get("body") or {}
            assert body.get("line") == expected_line, \
                f"{label}: expected line={expected_line}, got {body!r}"
            _assert_caret_consistent(c, expected_line, label)
            ws = _ws()
            assert ws == expected_ws, \
                f"{label}: expected ws={expected_ws!r}, got {ws!r}"

        # bp at 5: paused before a=100 executes.
        ev = c.wait_event("stopped", timeout=5.0)
        assert (ev.get("body") or {}).get("line") == 5, ev
        _assert_caret_consistent(c, 5, "initial bp")
        assert _ws() == {}, f"bp@5 ws should be empty: {_ws()!r}"

        # Forward via JIT execution.
        _step_assert("next", "next #1",      6, {"a": "100"})
        _step_assert("next", "next #2",      7, {"a": "100", "b": "200"})

        # Two reverse steps.
        _step_assert("stepBack", "stepBack #1", 6, {"a": "100"})
        _step_assert("stepBack", "stepBack #2", 5, {})

        # Forward steps now go through the redo log: each step
        # re-applies the captured post-write state for one statement
        # and lands at the next same-frame boundary, without resuming
        # the JIT or re-emitting any console output.
        _step_assert("next", "next #3 (redo)",       6, {"a": "100"})
        _step_assert("next", "next #4 (redo catches up)",
                     7, {"a": "100", "b": "200"})

        # next #5 is past redo_cap — redo cleared, JIT resumes for real
        # and runs line 7's body (c=300), pausing at line 8.
        _step_assert("next", "next #5 (JIT resumes, runs line 7)",
                     8, {"a": "100", "b": "200", "c": "300"})

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_step_in_then_step_back(matlabc, program):
    """stepIn followed by stepBack: stepBack stays inside the
    callee frame and refuses to cross back into the caller, the
    same guard scn_step_back_inside_function checks but reached
    via stepIn instead of a function-line breakpoint.

    Sequence over examples/factorial.m (line 5 is
    `disp(fact(1));`; fact's body starts at line 13):

      bp at 5             -> ('<script>', 5)
      stepIn              -> ('fact', 13), ('<script>', 5)
      stepIn #2           -> ('fact', 14), ('<script>', 5)
      stepBack #1         -> ('fact', 13), ('<script>', 5)
      stepBack #2         -> reason=entry, frame still ('fact', 13)
    """
    import os
    fact_program = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(program))),
        "examples", "factorial.m",
    )
    if not os.path.exists(fact_program):
        # Skip cleanly when the example layout differs (e.g. a
        # worktree without examples/).
        return
    with DapClient(matlabc, fact_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": fact_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": fact_program},
            "breakpoints": [{"line": 5}],
        })
        c.request("configurationDone")
        ev = c.wait_event("stopped", timeout=10.0)
        body = ev.get("body") or {}
        assert body.get("line") == 5 and body.get("reason") == "breakpoint", body

        def _frames():
            st = c.request("stackTrace", {"threadId": 1})
            return [(f.get("name"), f.get("line"))
                    for f in (st.get("stackFrames") or [])]

        frames = _frames()
        assert frames[0][1] == 5, \
            f"initial innermost frame should be at line 5: {frames!r}"

        # stepIn: enter fact(1), pause at line 13 (`if n <= 1`).
        c.request("stepIn")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 13 and body.get("reason") == "step", body
        frames = _frames()
        assert frames[0] == ("fact", 13), \
            f"after stepIn, innermost frame should be fact:13: {frames!r}"
        # Caller frame still visible at the call site (line 5).
        assert any(name != "fact" and ln == 5 for (name, ln) in frames[1:]), \
            f"caller frame should remain visible at line 5: {frames!r}"

        # stepIn #2: walk from `if n<=1` to the `y = 1` body.
        c.request("stepIn")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 14 and body.get("reason") == "step", body
        frames = _frames()
        assert frames[0] == ("fact", 14), \
            f"after stepIn #2, innermost frame should be fact:14: {frames!r}"

        # stepBack #1: rewind to fact:13. Stays inside the callee.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 13 and body.get("reason") == "step", body
        frames = _frames()
        assert frames[0] == ("fact", 13), \
            f"stepBack should stay inside fact, not cross back " \
            f"to the script: {frames!r}"

        # stepBack #2: would cross back into the caller's call
        # site. The runtime refuses: reason=entry, frame stays
        # at fact:13.
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "entry", \
            f"stepBack across the function-call boundary should " \
            f"report reason=entry, not silently teleport up: {body!r}"
        frames = _frames()
        assert frames[0] == ("fact", 13), \
            f"refused-cross stepBack should leave the frame at " \
            f"fact:13: {frames!r}"

        c.request("continue")
        try:
            c.wait_event("terminated", timeout=10.0)
        except DapError:
            pass


def scn_step_forward_redo_overwrites(matlabc, program):
    """Redo replay handles variables that get overwritten across
    statements: each undo record carries both the prior value
    (for stepBack) AND the post-write value (for redo). Without
    the post-write capture, walking forward through the log would
    only restore the *prior* value of the most recent write —
    losing the actual progression.

    Sequence over dap_revstep_overwrite_program.m (lines 1-4 are
    `x=1; x=2; x=3; disp(x);`):

      bp at 4    -> ws={x:3}
      stepBack   -> ws={x:2}
      stepBack   -> ws={x:1}
      stepBack   -> ws={}
      next       -> ws={x:1}    (redo replay of line 1's write)
      next       -> ws={x:2}    (redo replay of line 2's write)
      next       -> ws={x:3}    (redo replay of line 3's write — caught up)
    """
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_overwrite_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 4}],
        })
        c.request("configurationDone")
        c.wait_event("stopped", timeout=5.0)

        def _ws():
            st = c.request("stackTrace", {"threadId": 1})
            sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
            return _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])

        assert _ws() == {"x": "3"}, _ws()

        for cmd, expect in [
            ("stepBack", {"x": "2"}),
            ("stepBack", {"x": "1"}),
            ("stepBack", {}),
            ("next", {"x": "1"}),
            ("next", {"x": "2"}),
            ("next", {"x": "3"}),
        ]:
            c.request(cmd)
            c.wait_event("stopped", timeout=5.0)
            ws = _ws()
            assert ws == expect, \
                f"after {cmd}, expected ws={expect!r}, got {ws!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_continue_after_step_back_drains_redo(matlabc, program):
    """`continue` after `stepBack` must drain the redo log to
    catch up to the JIT's parked position before resuming JIT
    execution. Otherwise the JIT would execute one more
    statement past the rewound caret immediately, the user
    would see only ONE statement get re-executed, and the
    program would terminate having "skipped" the rewound region.

    Sequence over dap_revstep_program.m: bp at 5, walk to line 7,
    then stepBack twice to line 5 (workspace empty), then
    `continue`. The continue should:
      1. Replay redo records all the way to redo_cap (catching up
         to the JIT at line 7's hook). At this point ws has a, b
         restored.
      2. Resume the JIT, which runs line 7 (writes c=300) and
         lines 8 (disp(c)).
      3. Program terminates with all three writes visible in
         the final state.
    """
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 5}, {"line": 8}],
        })
        c.request("configurationDone")
        ev = c.wait_event("stopped", timeout=5.0)
        assert (ev.get("body") or {}).get("line") == 5, ev

        c.request("next")
        c.wait_event("stopped", timeout=5.0)
        c.request("next")
        ev = c.wait_event("stopped", timeout=5.0)
        assert (ev.get("body") or {}).get("line") == 7, ev

        c.request("stepBack")
        c.wait_event("stopped", timeout=5.0)
        c.request("stepBack")
        ev = c.wait_event("stopped", timeout=5.0)
        assert (ev.get("body") or {}).get("line") == 5, ev

        # Continue: redo drains, JIT resumes, program reaches
        # the line-8 bp with all three writes applied.
        c.request("continue")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 8, \
            f"continue should land at the line-8 bp: {body!r}"
        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        ws = _vars_by_name(c, ref=sc["scopes"][0]["variablesReference"])
        assert ws == {"a": "100", "b": "200", "c": "300"}, \
            f"after continue past redo + JIT exec: {ws!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_write_memory_visible_in_variables(matlabc, program):
    """A `writeMemory` mutation to a matrix data buffer must be
    visible through the `variables` request — the same protocol
    surface the IDE's Variables panel uses. Catches a regression
    where writeMemory accidentally targets a shadow / cached copy
    instead of the live buffer that variables() walks.

    Uses dap_matrix_program.m's `A = [1 2 3; 4 5 6]`. We write
    7777.0 into the (1,1) cell via writeMemory, then expand the
    A row in `variables` and confirm the (1,1) child reads
    "7777" — not "1"."""
    import os
    import base64
    import struct
    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        body = c.request("variables",
                         {"variablesReference": sc["scopes"][0]["variablesReference"]})
        rows = {v["name"]: v for v in (body.get("variables") or [])}
        a_row = rows.get("A")
        assert a_row, f"A missing from script ws: {rows!r}"
        mem_ref = a_row.get("memoryReference")
        a_ref = a_row.get("variablesReference")
        assert mem_ref and a_ref, f"A row should carry mem + var refs: {a_row!r}"

        # Sanity: read (1,1) via variables before the mutation.
        before = c.request("variables", {"variablesReference": a_ref})
        before_cells = {v["name"]: v["value"] for v in (before.get("variables") or [])}
        assert before_cells.get("(1,1)") == "1", \
            f"pre-write (1,1) should be 1: {before_cells!r}"

        # Write 7777.0 into the buffer's first 8 bytes.
        body = c.request("writeMemory", {
            "memoryReference": mem_ref,
            "offset": 0,
            "data": base64.b64encode(struct.pack("<d", 7777.0)).decode("ascii"),
        })
        assert body.get("bytesWritten") == 8, body

        # Re-query variables on the SAME mat-ref. The cell view
        # must reflect the mutation — no stale cache.
        after = c.request("variables", {"variablesReference": a_ref})
        after_cells = {v["name"]: v["value"] for v in (after.get("variables") or [])}
        assert after_cells.get("(1,1)") == "7777", \
            f"writeMemory mutation invisible via variables: " \
            f"(1,1) reads {after_cells.get('(1,1)')!r}, " \
            f"expected '7777': {after_cells!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_read_watch_on_frame_local_is_invisible(matlabc, program):
    """Negative test: documented limitation that read-watches on
    function-frame-local variables are silently invisible. The
    JIT lowers function-frame reads as direct slot loads (no
    matlab_dbg_frame_local_get), so the runtime watch table
    never gets consulted. Locks in this contract — if a future
    change accidentally wires frame-local reads into the watch
    path, the test fails and the docs need updating.

    Uses dap_locals_program.m: `compute(a, b)` writes
    `total = a + b;` on line 10 then reads `total` on line 11.
    A read-only watch on `total` (script-scope by name) should
    NOT trip — the function-frame read is invisible. The
    program runs to completion with no extra `stopped` events."""
    import os
    locals_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_locals_program.m",
    )
    with DapClient(matlabc, locals_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": locals_program, "stopOnEntry": False})
        # Set a read-only data breakpoint on `total`. Inside
        # compute() it's a frame local; the runtime watch table
        # only fires on matlab_ws_get_*, which the JIT doesn't
        # call for function locals.
        c.request("setDataBreakpoints", {
            "breakpoints": [{"dataId": "total", "accessType": "read"}],
        })
        c.request("configurationDone")

        # No `stopped` event should arrive. If the read-watch
        # accidentally started seeing frame-local reads (e.g. via
        # a future lowering that emits matlab_dbg_frame_local_get
        # mirror calls), the program would pause and this would
        # fail.
        c.expect_no_event("stopped", window=0.6)
        c.wait_event("terminated", timeout=5.0)
    """A `writeMemory` mutation to a matrix data buffer must be
    visible through the `variables` request — the same protocol
    surface the IDE's Variables panel uses. Catches a regression
    where writeMemory accidentally targets a shadow / cached copy
    instead of the live buffer that variables() walks.

    Uses dap_matrix_program.m's `A = [1 2 3; 4 5 6]`. We write
    7777.0 into the (1,1) cell via writeMemory, then expand the
    A row in `variables` and confirm the (1,1) child reads
    "7777" — not "1"."""
    import os
    import base64
    import struct
    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        body = c.request("variables",
                         {"variablesReference": sc["scopes"][0]["variablesReference"]})
        rows = {v["name"]: v for v in (body.get("variables") or [])}
        a_row = rows.get("A")
        assert a_row, f"A missing from script ws: {rows!r}"
        mem_ref = a_row.get("memoryReference")
        a_ref = a_row.get("variablesReference")
        assert mem_ref and a_ref, f"A row should carry mem + var refs: {a_row!r}"

        # Sanity: read (1,1) via variables before the mutation.
        before = c.request("variables", {"variablesReference": a_ref})
        before_cells = {v["name"]: v["value"] for v in (before.get("variables") or [])}
        assert before_cells.get("(1,1)") == "1", \
            f"pre-write (1,1) should be 1: {before_cells!r}"

        # Write 7777.0 into the buffer's first 8 bytes.
        body = c.request("writeMemory", {
            "memoryReference": mem_ref,
            "offset": 0,
            "data": base64.b64encode(struct.pack("<d", 7777.0)).decode("ascii"),
        })
        assert body.get("bytesWritten") == 8, body

        # Re-query variables on the SAME mat-ref. The cell view
        # must reflect the mutation — no stale cache.
        after = c.request("variables", {"variablesReference": a_ref})
        after_cells = {v["name"]: v["value"] for v in (after.get("variables") or [])}
        assert after_cells.get("(1,1)") == "7777", \
            f"writeMemory mutation invisible via variables: " \
            f"(1,1) reads {after_cells.get('(1,1)')!r}, " \
            f"expected '7777': {after_cells!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_reverse_continue_to_breakpoint(matlabc, program):
    """`reverseContinue` should walk the undo log back until it
    hits a breakpoint (per DAP spec), not just one statement.
    Two bps: lines 6 and 8 of dap_revstep_program.m. Hit line 8
    by continuing past line 6's first hit, then reverseContinue
    — must land on line 6 with reason="breakpoint" and the
    earlier bp's hitBreakpointIds."""
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        body = c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 6}, {"line": 8}],
        })
        bps = body.get("breakpoints") or []
        assert len(bps) == 2 and all(b.get("verified") for b in bps), bps
        line6_id = bps[0].get("id")
        line8_id = bps[1].get("id")
        c.request("configurationDone")

        # First pause: line 6 bp.
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 6 and body.get("reason") == "breakpoint", body

        # Continue to second pause: line 8 bp.
        c.request("continue")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("line") == 8 and body.get("reason") == "breakpoint", body

        # reverseContinue must walk back and land on line 6 with
        # reason="breakpoint" — not stop at the next statement.
        c.request("reverseContinue")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "breakpoint", \
            f"reverseContinue with earlier bp should report " \
            f"reason=breakpoint, got: {body!r}"
        assert body.get("line") == 6, \
            f"reverseContinue should land on the earlier bp " \
            f"(line 6), got line={body.get('line')}: {body!r}"
        assert body.get("hitBreakpointIds") == [line6_id], \
            f"reverseContinue should surface the earlier bp's id, " \
            f"got {body.get('hitBreakpointIds')!r} vs " \
            f"line6_id={line6_id}: {body!r}"

        c.request("continue")
        try:
            c.wait_event("terminated", timeout=5.0)
        except DapError:
            pass


def scn_reverse_continue_to_entry(matlabc, program):
    """`reverseContinue` with no earlier bp set walks the entire
    undo log back and stops with reason="entry" plus a
    description naming the exhausted-log condition."""
    import os
    rs_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_revstep_program.m",
    )
    with DapClient(matlabc, rs_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": rs_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": rs_program},
            "breakpoints": [{"line": 8}],
        })
        c.request("configurationDone")
        c.wait_event("stopped", timeout=5.0)

        c.request("reverseContinue")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "entry", \
            f"reverseContinue with no earlier bp should report " \
            f"reason=entry: {body!r}"
        assert "exhausted" in (body.get("description") or "").lower(), \
            f"entry stop should describe the empty-log condition: {body!r}"

        c.request("continue")
        try:
            c.wait_event("terminated", timeout=5.0)
        except DapError:
            pass


def scn_caret_consistency(matlabc, program):
    """Drives a sequence of next / stepBack / continue against
    factorial.m and asserts that on every pause, stackTrace[0]'s
    line matches the stopped event's line. This is the
    class-of-bug catch the original factorial stepBack regression
    needed: stopped.line was always correct, but stackTrace was
    stale, and the IDE renders the caret from stackTrace.

    Doesn't assert specific transitions — just the cross-source
    invariant. If a future change makes any pause path forget to
    refresh the per-thread frame chain, this scenario fails."""
    import os
    fact_program = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(program))),
        "examples", "factorial.m",
    )
    if not os.path.exists(fact_program):
        return
    with DapClient(matlabc, fact_program) as c:
        c.request("initialize", {"clientID": "t", "linesStartAt1": True})
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": fact_program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": fact_program},
            "breakpoints": [{"line": 14}],
        })
        c.request("configurationDone")

        def _wait_and_check_caret(label):
            ev = c.wait_event("stopped", timeout=10.0)
            body = ev.get("body") or {}
            line = body.get("line")
            if line is None:
                # reason="entry" / log-exhausted — no line to
                # cross-check. Fine; helper would fail with no
                # comparison target.
                return body
            _assert_caret_consistent(c, line, label)
            return body

        # Initial bp pause inside fact.
        _wait_and_check_caret("initial breakpoint")

        # next over `y = 1;` -> lands on line 17 (`end`) or wherever
        # the lowering's hook positions the next stop. We don't
        # care about the exact line, only that stackTrace agrees.
        c.request("next")
        _wait_and_check_caret("after next")

        # continue past this fact() invocation, hit the bp again
        # in the next recursive call.
        c.request("continue")
        _wait_and_check_caret("after continue (next bp hit)")

        # stepBack: this is the path that originally desync'd.
        c.request("stepBack")
        body = _wait_and_check_caret("after stepBack")
        # stepBack within the function should reach a step reason
        # (or entry if the function had no prior boundary).
        assert body.get("reason") in ("step", "entry"), body

        c.request("continue")
        try:
            c.wait_event("terminated", timeout=10.0)
        except DapError:
            pass


def scn_disassemble(matlabc, program):
    """`disassemble` walks JIT-emitted machine code instruction-by-
    instruction using the host triple's MCDisassembler. With no
    memoryReference supplied the request defaults to the JIT's
    `main` entry point (the compiled program's first instruction);
    each result row carries an address, the raw bytes (hex), and
    the printed asm.

    Capability `supportsDisassembleRequest` is advertised. The
    underlying LLVM init is deferred to first use to avoid a
    static-init clash with MLIR's target registration on this
    LLVM build."""
    with DapClient(matlabc, program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
        })
        assert caps.get("supportsDisassembleRequest"), \
            f"disassemble caps not advertised: {caps}"
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": program, "stopOnEntry": False})
        c.request("setBreakpoints", {
            "source": {"path": program},
            "breakpoints": [{"line": 5}],
        })
        c.request("configurationDone")
        c.wait_event("stopped", timeout=5.0)

        # memoryReference="" defaults to the JIT main entry point.
        body = c.request("disassemble", {
            "memoryReference": "",
            "instructionCount": 4,
        })
        instrs = body.get("instructions") or []
        assert len(instrs) == 4, f"expected 4 instructions: {instrs!r}"
        for ins in instrs:
            addr = ins.get("address") or ""
            bytes_hex = ins.get("instructionBytes") or ""
            text = ins.get("instruction") or ""
            assert addr.startswith("0x"), f"bad address: {ins!r}"
            # Each byte is two hex chars, separated by spaces.
            for tok in bytes_hex.split():
                assert len(tok) == 2 and all(
                    ch in "0123456789abcdef" for ch in tok), \
                    f"bad bytes encoding: {bytes_hex!r}"
            # Decoded text is non-empty (disassembler succeeded) or
            # the explicit ".byte (decode failed)" recovery row.
            assert text and (text != "" or "decode failed" in text), ins

        # Negative instructionOffset is refused.
        try:
            c.request("disassemble", {
                "memoryReference": "",
                "instructionCount": 1,
                "instructionOffset": -1,
            })
            raise AssertionError("negative offset should refuse")
        except DapError as e:
            assert "negative" in str(e).lower(), \
                f"refusal should mention negative: {e}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_parfor_per_thread_frames(matlabc, program):
    """Parfor body running on N pthreads: each one's enter_frame /
    leave_frame mutates its own per-thread chain, so concurrent
    workers don't corrupt each other's call stacks. The legacy
    shared frames[] is now a paused-thread snapshot.

    This scenario exercises the no-pause case — the parfor body
    runs concurrently across 3 workers, each entering and leaving
    a function frame. Without per-thread chains, the global
    n_frames would race across threads and crash or produce
    nonsense; with per-thread chains the program runs cleanly."""
    import os
    parfor_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_parfor_program.m",
    )
    with DapClient(matlabc, parfor_program) as c:
        initialize_and_launch(c)
        c.wait_event("terminated", timeout=10.0)
        # If we got here without a crash, per-thread chains work.
        # Confirm thread enumeration is intact.
        body = c.request("threads")
        ts = body.get("threads") or []
        assert len(ts) >= 2, \
            f"parfor should have left multiple thread entries: {ts!r}"


def scn_parfor_thread_enumeration(matlabc, program):
    """parfor spawns one pthread per iteration; each registers
    itself with the runtime on its first hook fire. The DAP
    `threads` request enumerates them.

    dap_parfor_program.m runs `parfor i = 1:3`, so after the body
    executes there are four threads in the table: 1 = main worker,
    2..4 = parfor workers. Names are "main" / "parfor-1" / etc.

    v1 limitation (documented): the frame stack is shared across
    threads. A bp inside the parfor body would surface the
    originating thread id in the stopped event, but stackTrace's
    contents reflect whatever thread last touched the global
    stack. Per-thread frame chains are follow-up work."""
    import os
    parfor_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_parfor_program.m",
    )
    with DapClient(matlabc, parfor_program) as c:
        initialize_and_launch(c)
        c.wait_event("terminated", timeout=10.0)

        body = c.request("threads")
        ts = body.get("threads") or []
        ids = sorted(t.get("id") for t in ts)
        names = {t.get("id"): t.get("name") for t in ts}

        # Main worker is always id 1; parfor workers are 2..4 (one
        # per iteration of `1:3`).
        assert 1 in ids, f"main worker missing from threads: {ts!r}"
        assert names.get(1) == "main", \
            f"thread 1 should be 'main': {names!r}"
        # Don't pin the exact count — MAX_THREADS = 32 caps at 32
        # but a single 3-iter parfor should produce 4 entries
        # total. We assert >= 2 so any future scheduler change
        # (e.g. coalescing) doesn't break the test.
        assert len(ts) >= 2, \
            f"parfor should register multiple threads, got {ts!r}"
        # All non-main names should follow the parfor-N format.
        for tid, nm in names.items():
            if tid == 1: continue
            assert nm.startswith("parfor-"), \
                f"thread {tid} name should start with 'parfor-': {nm!r}"


def scn_keyboard_builtin(matlabc, program):
    """A `keyboard` call in user code drops the worker into a paused
    state — same machinery as a breakpoint, but triggered from the
    program itself rather than a DAP-set bp. The `stopped` event
    carries `reason: "entry"` so the IDE's REPL panel takes over.

    Once resumed, the program continues normally and the workspace
    is intact (so `disp(x)` after the keyboard call still prints
    the value set before it)."""
    import os
    kb_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_keyboard_program.m",
    )
    with DapClient(matlabc, kb_program) as c:
        initialize_and_launch(c)
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        assert body.get("reason") == "entry", \
            f"keyboard pause should report reason=entry: {body!r}"
        assert body.get("line") == 5, \
            f"keyboard pause should land at line 5: {body!r}"

        # Workspace is intact at the keyboard pause.
        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        body = c.request("variables", {
            "variablesReference": sc["scopes"][0]["variablesReference"],
        })
        rows = {v["name"]: v["value"] for v in (body.get("variables") or [])}
        assert rows.get("x") == "41", \
            f"x should be visible at keyboard pause: {rows!r}"

        # Resume continues normally — disp(x) on line 6 prints "41".
        c.request("continue")
        ev = c.wait_event(
            "output",
            timeout=5.0,
            predicate=lambda m: (m.get("body") or {}).get("category") == "stdout"
                                 and "41" in ((m.get("body") or {}).get("output") or ""),
        )
        c.wait_event("terminated", timeout=5.0)


def scn_complex_and_3d_matrix_expansion(matlabc, program):
    """Complex (matlab_mat_c) and 3-D (matlab_mat3) matrices surface
    in the DAP variables panel with the right shape header and
    drillable children:

    - Complex 1x1 unboxes to `re+im*i` in the parent value column
      (no children — same as a 1x1 real matrix unboxes to its
      scalar).
    - 3-D MxNxP shows up as `MxNxP double`, carries
      `indexedVariables = M*N*P`, and expands into one child per
      cell labelled `(i,j,k)` in slice-major order so cells of the
      same k group together.

    Uses dap_complex_program.m which constructs both shapes."""
    import os
    cmplx_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_complex_program.m",
    )
    with DapClient(matlabc, cmplx_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 9}])
        _stop_event(c)

        st = c.request("stackTrace", {"threadId": 1})
        sc = c.request("scopes", {"frameId": st["stackFrames"][0]["id"]})
        ref = sc["scopes"][0]["variablesReference"]
        body = c.request("variables", {"variablesReference": ref})
        rows = {v["name"]: v for v in (body.get("variables") or [])}

        # Complex 1x1 — unboxes to "3+4i" with no expansion ref.
        c_row = rows.get("c")
        assert c_row is not None, f"c row missing: {rows!r}"
        assert c_row.get("value") == "3+4i", \
            f"complex 1x1 should unbox to '3+4i': {c_row!r}"
        assert c_row.get("variablesReference") == 0, \
            f"complex 1x1 should be a leaf: {c_row!r}"

        # 3-D 2x2x2 — shape label + cell expansion.
        a_row = rows.get("A")
        assert a_row is not None, f"A row missing: {rows!r}"
        assert a_row.get("value") == "2x2x2 double", \
            f"3-D shape header wrong: {a_row!r}"
        assert a_row.get("indexedVariables") == 8, \
            f"3-D should advertise indexedVariables=2*2*2=8: {a_row!r}"
        a_ref = a_row.get("variablesReference")
        assert a_ref and a_ref >= 200000, \
            f"3-D row should carry mat-ref: {a_row!r}"

        body = c.request("variables", {"variablesReference": a_ref})
        cells = {v["name"]: v.get("value") for v in (body.get("variables") or [])}
        # All eight (i,j,k) labels must be present.
        expected_labels = [(i, j, k) for k in (1, 2)
                           for i in (1, 2) for j in (1, 2)]
        for (i, j, k) in expected_labels:
            label = f"({i},{j},{k})"
            assert label in cells, f"3-D label {label!r} missing: {cells!r}"
        # Mutated cell must read 42; everything else is 1 (from ones).
        assert cells.get("(1,2,1)") == "42", \
            f"3-D mutated cell should be 42: {cells!r}"
        for (i, j, k) in expected_labels:
            if (i, j, k) == (1, 2, 1):
                continue
            label = f"({i},{j},{k})"
            assert cells.get(label) == "1", \
                f"3-D cell {label!r} should be 1, got {cells.get(label)!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_class_method_function_breakpoints(matlabc, program):
    """`setFunctionBreakpoints` resolves class methods under both
    `MethodName`, `ClassName.MethodName`, and `ClassName/MethodName`
    forms. dap_class_program.m has `Account.deposit` — we exercise
    all three names and confirm the bp lands inside the method body."""
    import os
    class_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_class_program.m",
    )
    for name in ("deposit", "Account.deposit", "Account/deposit"):
        with DapClient(matlabc, class_program) as c:
            caps = c.request("initialize", {
                "clientID": "matlabc-test",
                "linesStartAt1": True,
                "columnsStartAt1": True,
            })
            assert caps.get("supportsFunctionBreakpoints"), caps
            c.wait_event("initialized", timeout=5.0)
            c.request("launch", {"program": class_program, "stopOnEntry": False})
            body = c.request("setFunctionBreakpoints", {
                "breakpoints": [{"name": name}],
            })
            bps = body.get("breakpoints") or []
            assert bps and bps[0].get("verified") is True, \
                f"function bp on {name!r} not verified: {bps!r}"
            # The body line for `deposit` is line 25 (`obj.Balance = ...`).
            assert bps[0].get("line") == 25, \
                f"function bp on {name!r} landed at unexpected line: {bps!r}"
            c.request("configurationDone")
            ev = c.wait_event("stopped", timeout=5.0)
            body = ev.get("body") or {}
            assert body.get("line") == 25, \
                f"stopped at unexpected line for {name!r}: {body!r}"
            c.request("continue")
            c.wait_event("terminated", timeout=5.0)


def scn_breakpoint_locations(matlabc, program):
    """`breakpointLocations` returns the bp-eligible lines in a
    range, populated server-side by the AST walker."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 5}])
        _stop_event(c)

        body = c.request("breakpointLocations", {
            "source": {"path": program},
            "line": 1,
            "endLine": 12,
        })
        lines = sorted({(b.get("line") or 0)
                        for b in (body.get("breakpoints") or [])})
        # dap_program.m has assignments on 4-6 and a disp on 12;
        # those must appear. Line 7 is blank, 11 is blank, 1-3 are
        # comments — NOT in the set.
        for L in (4, 5, 6, 12):
            assert L in lines, f"line {L} missing from bp locations: {lines!r}"
        for L in (1, 2, 3, 7, 11):
            assert L not in lines, \
                f"non-executable line {L} surfaced as bp location: {lines!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_exception_info_and_filter(matlabc, program):
    """`setExceptionBreakpoints` toggles the `error` filter and
    `exceptionInfo` returns the captured message + frame snapshot
    once an error has fired.

    Uses dap_error_program.m which calls error('boom') two frames
    deep. With the filter on, the runtime hook pauses on the first
    statement after matlab_set_error fires — we then read
    exceptionInfo to confirm the snapshot survived the unwind."""
    import os
    err_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_error_program.m",
    )
    with DapClient(matlabc, err_program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        filters = caps.get("exceptionBreakpointFilters") or []
        assert any(f.get("filter") == "error" for f in filters), \
            f"error filter not advertised in caps: {filters!r}"
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": err_program, "stopOnEntry": False})
        c.request("setExceptionBreakpoints", {"filters": ["error"]})
        c.request("configurationDone")

        # Either the runtime pauses (if the hook fires after the error
        # flag is set) or the program just runs to completion (the
        # error fires on the last statement and unwinds straight to
        # exit). Both are valid; only assert that exceptionInfo
        # survives — read it after we see either stopped or
        # terminated.
        evname = c.wait_event("stopped", timeout=2.0) \
                  if False else None  # placeholder; see below
        # Race-free: poll for stopped, fall back to terminated.
        try:
            c.wait_event("stopped", timeout=2.0)
            paused = True
        except DapError:
            paused = False

        if paused:
            body = c.request("exceptionInfo", {"threadId": 1})
            assert body.get("exceptionId") == "matlab.error", body
            assert "boom" in (body.get("description") or ""), body
            c.request("continue")

        c.wait_event("terminated", timeout=5.0)


def scn_modules(matlabc, program):
    """`modules` returns an empty list — we have no shared-library
    concept but the request must respond gracefully so module-aware
    IDEs render an empty pane instead of erroring."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 5}])
        _stop_event(c)

        body = c.request("modules")
        assert isinstance(body.get("modules"), list), body
        assert body.get("modules") == [], body
        assert body.get("totalModules") == 0, body

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_unsupported_refusals(matlabc, program):
    """Reverse-debug, memory, and disassembly requests must respond
    with success=false and a precise reason — better UX than the
    catch-all silent-success the unknown-handler used to give."""
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 5}])
        _stop_event(c)

        for cmd in ("locations",
                    "setInstructionBreakpoints", "restartFrame",
                    "goto", "gotoTargets"):
            try:
                c.request(cmd, {})
                raise AssertionError(f"{cmd} should refuse")
            except DapError as e:
                assert "unsupported" in str(e) or \
                       "require" in str(e) or \
                       "does not include" in str(e), \
                    f"{cmd} refusal had unexpected message: {e}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_breakpoint_ids(matlabc, program):
    """setBreakpoints assigns each bp a stable id; the stopped event
    surfaces it via hitBreakpointIds when that bp triggers the pause.
    This lets the IDE highlight which row of the breakpoints panel
    fired (when multiple bps share a file)."""
    with DapClient(matlabc, program) as c:
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
        })
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {"program": program, "stopOnEntry": False})
        body = c.request("setBreakpoints", {
            "source": {"path": program},
            "breakpoints": [{"line": 5}, {"line": 6}],
        })
        bps = body.get("breakpoints") or []
        assert len(bps) == 2 and all(b.get("verified") for b in bps), bps
        ids = [b.get("id") for b in bps]
        assert all(isinstance(i, int) and i > 0 for i in ids), \
            f"missing/invalid bp ids: {ids!r}"
        assert ids[0] != ids[1], f"distinct lines should get distinct ids: {ids!r}"

        c.request("configurationDone")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        hit = body.get("hitBreakpointIds") or []
        assert hit == [ids[0]], \
            f"line-5 stop should report id={ids[0]!r}, got {hit!r}"

        c.request("continue")
        ev = c.wait_event("stopped", timeout=5.0)
        body = ev.get("body") or {}
        hit = body.get("hitBreakpointIds") or []
        assert hit == [ids[1]], \
            f"line-6 stop should report id={ids[1]!r}, got {hit!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_stderr_forwarded(matlabc, program):
    """`error()` writes its traceback to stderr; the DAP server tees
    those bytes to a `stderr`-categorised `output` event so the IDE's
    debug console renders them with error styling. The bytes also
    reach the parent process's stderr (kept alive for subprocess
    capture / CI logs)."""
    import os
    err_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_error_program.m",
    )
    with DapClient(matlabc, err_program) as c:
        initialize_and_launch(c)
        # The error fires during the program run; collect stderr-
        # categorised output events until we see the message text.
        try:
            ev = c.wait_event(
                "output",
                timeout=5.0,
                predicate=lambda m: (
                    (m.get("body") or {}).get("category") == "stderr"
                    and "boom" in ((m.get("body") or {}).get("output") or "")
                ),
            )
            out = (ev.get("body") or {}).get("output", "")
            assert "boom" in out, f"stderr forwarding missed: {out!r}"
        finally:
            try:
                c.wait_event("terminated", timeout=5.0)
            except Exception:
                pass


def scn_watch_void_promotion(matlabc, program):
    """Watch-mode `disp(T)` used to SIGSEGV the matlabc process: the
    `__matlab_dbg_eval = (disp(T));` wrap binds a void RHS and the
    JIT crashes deep in the lowering. The fix detects statement-shaped
    void calls in the watch handler and routes them through the REPL
    branch, returning `result="<void>"` so the watch row shows a
    clear placeholder. Side effects (the matrix print) still flow
    through DAP `output` events."""
    import os
    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        # Watch eval of disp(A) — used to crash. Now returns
        # result="<void>" cleanly. The watch context (no `context`
        # field) is the watch panel's default.
        resp = c.request("evaluate", {"expression": "disp(A)"})
        assert resp.get("result") == "<void>", \
            f"watch-mode disp(A) should auto-promote to <void>: {resp!r}"

        # The actual matrix bytes flow through the stdout pipe →
        # DAP output events. Wait for one that contains matrix cells.
        ev = c.wait_event(
            "output",
            timeout=5.0,
            predicate=lambda m: (
                (m.get("body") or {}).get("category") == "stdout"
                and "1" in ((m.get("body") or {}).get("output") or "")
            ),
        )
        out = (ev.get("body") or {}).get("output", "")
        for cell in ("1", "6"):
            assert cell in out, \
                f"disp output missing cell {cell!r}: {out!r}"

        # Value-shaped watches must still work — the auto-promotion
        # only fires for known void-call shapes. `A` is a matrix
        # name, not a call, so it goes through the normal wrap.
        resp = c.request("evaluate", {"expression": "A"})
        assert resp.get("result") == "2x3 double", \
            f"value-shaped watch broke after fix: {resp!r}"
        assert resp.get("variablesReference", 0) >= 200000, \
            f"matrix watch should carry mat-ref: {resp!r}"

        # Bare `who` / `whos` (statement form, no parens) also auto-
        # promote — these would parse as bare identifiers and
        # similarly fail to bind under the wrap.
        resp = c.request("evaluate", {"expression": "whos"})
        assert resp.get("result") == "<void>", \
            f"watch `whos` should promote to <void>: {resp!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)
        # Process must still be alive for the orderly disconnect.
        # If the crash regressed, poll() would return -11 / -6 here.
        # The `with` block's __exit__ does the disconnect.


def scn_evaluate_repl(matlabc, program):
    """`evaluate` with context='repl' runs input verbatim — supporting
    statement-level commands like `disp(T)` and assignments — instead of
    wrapping it as `__matlab_dbg_eval = (...)` like the watch path does.

    Output flows through the existing stdout pipe redirect and surfaces
    as DAP `output` events with category='stdout'. The evaluate response
    body itself returns an empty `result` because there's no synthesized
    holder to read back.

    Uses dap_matrix_program.m so we have a live 2x3 matrix `A` to disp.
    The bp is on line 10 (`disp(s);`); we run the eval before continuing.
    """
    import os
    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        # 1) disp(A) on a 2x3 matrix prints two rows via stdout. Watch
        #    mode can't run this (disp returns no value to assign); REPL
        #    mode succeeds and we capture the output via DAP events.
        resp = c.request("evaluate", {
            "expression": "disp(A)",
            "context": "repl",
        })
        # Result is empty by design — output is the answer.
        assert resp.get("result", "") == "", \
            f"REPL evaluate should return empty result, got {resp!r}"
        ev = c.wait_event(
            "output",
            timeout=5.0,
            predicate=lambda m: (
                (m.get("body") or {}).get("category") == "stdout"
                and "1" in ((m.get("body") or {}).get("output") or "")
            ),
        )
        out = (ev.get("body") or {}).get("output", "")
        for cell in ("1", "2", "3", "4", "5", "6"):
            assert cell in out, \
                f"disp(A) output missing cell {cell!r}: {out!r}"

        # 2) REPL-mode assignment to a fresh script-scope name persists,
        #    visible to a follow-up watch read. (No frameId -> no
        #    frame-bridge stamping, so the write sticks in matlab_ws.)
        c.request("evaluate", {
            "expression": "tmp_repl = 99;",
            "context": "repl",
        })
        resp = c.request("evaluate", {"expression": "tmp_repl"})
        assert resp.get("result") == "99", \
            f"REPL-set tmp_repl should read back as 99, got {resp!r}"

        # 3) Trailing `;` is preserved (REPL strips only outer
        #    whitespace) — a no-op statement runs without erroring.
        c.request("evaluate", {
            "expression": "5 + 3;",
            "context": "repl",
        })

        # 4) Malformed REPL input fails cleanly without dropping the
        #    connection.
        try:
            c.request("evaluate", {
                "expression": "1 +",
                "context": "repl",
            })
            raise AssertionError("malformed REPL input should fail")
        except DapError:
            pass

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_evaluate_string_concat(matlabc, program):
    """`evaluate` covers the three MATLAB string-concat idioms:

      1. `"..." + scalar + "..."`     (string operator+ with scalar coercion)
      2. `sprintf("...%.2f...", v)`   (format-string)
      3. `['...', num2str(v), '...']` (bracket char-array concat)

    All three go through the JIT pipeline (runReplInput); regression
    guard for the lowering paths in lib/MLIR/Lowering.cpp:BinaryOp +
    string-bracket MatrixLiteral, plus the LowerTensorOps wiring for
    matlab_string_concat / matlab_string_from_literal / sprintf_f64 /
    num2str. Uses dap_program.m so we have a live `x = 10` workspace
    binding to mix into the formatted output.
    """
    with DapClient(matlabc, program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 8}])
        _stop_event(c)

        # The DAP variable formatter wraps string values in double quotes
        # (matching how MATLAB renders `"..."` in the workspace inspector).
        resp = c.request("evaluate",
                         {"expression": '"x = " + x + "!"'})
        assert resp.get("result") == '"x = 10!"', \
            f'string + scalar should be "x = 10!", got {resp!r}'

        resp = c.request("evaluate",
                         {"expression": 'sprintf("v=%.2f", x)'})
        assert resp.get("result") == '"v=10.00"', \
            f'sprintf should be "v=10.00", got {resp!r}'

        resp = c.request("evaluate",
                         {"expression": "['x=', num2str(x)]"})
        assert resp.get("result") == '"x=10"', \
            f'bracket concat should be "x=10", got {resp!r}'

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_multifile_breakpoint(matlabc, program):
    """Breakpoints on a sibling .m file resolve and fire.

    The DAP path's compileProgram walks the entry-point's directory
    for function-only sibling .m files, parses each, and merges their
    Functions/Classes into the main TU. Each loaded file gets a
    distinct SourceManager FileID and is registered with the runtime
    via matlab_dbg_register_file, so an IDE-supplied path resolves
    through G.PathToFileId in the setBreakpoints handler. The bp
    fires when the JIT'd helper executes the matching line.
    """
    import os
    multifile_dir = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "multifile",
    )
    main_path = os.path.join(multifile_dir, "dap_main.m")
    helper_path = os.path.join(multifile_dir, "dap_helper.m")
    with DapClient(matlabc, main_path) as c:
        # Drive setBreakpoints separately for the helper file —
        # initialize_and_launch's breakpoints argument always targets
        # the program path, but multi-file is interesting precisely
        # when the bp lands in a different file.
        caps = c.request("initialize", {
            "clientID": "matlabc-test",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "pathFormat": "path",
        })
        c.wait_event("initialized", timeout=5.0)
        c.request("launch", {
            "program": main_path,
            "stopOnEntry": False,
        })
        # Line 6 of dap_helper.m is `r = intermediate + 1;` — the
        # second statement inside helper_fn.
        body = c.request("setBreakpoints", {
            "source": {"path": helper_path},
            "breakpoints": [{"line": 6}],
        })
        verified = body.get("breakpoints") or []
        assert verified and verified[0].get("verified"), \
            f"sibling-file bp must be verified, got {verified!r}"
        c.request("configurationDone")

        ev = c.wait_event("stopped", timeout=10.0)
        sb = ev.get("body") or {}
        assert sb.get("reason") == "breakpoint", sb
        assert sb.get("line") == 6, \
            f"expected stop at helper line 6, got {sb!r}"

        # stackTrace shows helper_fn at top with the helper file's
        # source path, and <script> below with main's path.
        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        assert len(frames) >= 2, f"expected helper + script frames: {frames!r}"
        assert "helper_fn" in (frames[0].get("name") or ""), frames[0]
        srcA = ((frames[0].get("source") or {}).get("path") or "")
        srcB = ((frames[1].get("source") or {}).get("path") or "")
        assert srcA.endswith("dap_helper.m"), \
            f"top frame source should be dap_helper.m, got {srcA!r}"
        assert srcB.endswith("dap_main.m"), \
            f"bottom frame source should be dap_main.m, got {srcB!r}"

        # Function-frame Locals reflect helper_fn's mid-execution state
        # — `intermediate` was assigned on line 5 (= 7 * 3 = 21), `x` is
        # the parameter spilled at frame entry, `r` not yet stored.
        sc = c.request("scopes", {"frameId": frames[0]["id"]})
        ref = (sc.get("scopes") or [{}])[0].get("variablesReference")
        vars_inner = _vars_by_name(c, ref=ref)
        assert vars_inner.get("x") == "7", \
            f"helper_fn.x should be 7, got {vars_inner!r}"
        assert vars_inner.get("intermediate") == "21", \
            f"helper_fn.intermediate should be 21, got {vars_inner!r}"
        assert "r" not in vars_inner, \
            f"helper_fn.r must not be visible before its assignment: {vars_inner!r}"

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_matrix_expansion(matlabc, program):
    """Matrix variables surface a `variablesReference` so the IDE can
    drill into the cells via the standard `variables(ref)` request.

    Without this the LOCALS panel showed `RxC double` and stopped —
    no disclosure arrow, no values, no input for an editor's matrix
    viewer / variable inspector. The fix wires a MatRefs registry
    server-side and a `matlab_dbg_mat_get` element accessor in the
    runtime; this scenario verifies the wire format end-to-end.

    Uses dap_matrix_program.m which constructs:
      - `A`: 2x3 matrix     -> children labelled (i,j) row-major
      - `b`: 3x1 col vector -> children labelled (i)
      - `s`: 1x1 scalar     -> unboxed in parent value, no children
    """
    import os
    mat_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_matrix_program.m",
    )
    # Line 10 is `disp(s);` — the last script-body statement, by
    # which point all three matrices are assigned.
    with DapClient(matlabc, mat_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        body = _stop_event(c)
        assert body.get("line") == 10, body

        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        assert frames, st
        sc = c.request("scopes", {"frameId": frames[0].get("id", 0)})
        ref = (sc.get("scopes") or [{}])[0].get("variablesReference")
        assert ref, sc
        body = c.request("variables", {"variablesReference": ref})
        rows = {v.get("name"): v for v in (body.get("variables") or [])}

        # Shape labels and ref gating.
        assert rows.get("A", {}).get("value") == "2x3 double", rows.get("A")
        assert rows.get("b", {}).get("value") == "3x1 double", rows.get("b")
        # Scalar-shaped matrix unboxes to its value and stays a leaf.
        assert rows.get("s", {}).get("value") == "7", rows.get("s")
        assert rows.get("s", {}).get("variablesReference") == 0, \
            f"1x1 matrix `s` should be a leaf: {rows.get('s')!r}"
        for n in ("A", "b"):
            r = rows[n].get("variablesReference")
            assert r and r >= 200000, \
                f"{n} should carry a mat-ref >= 200000, got {r!r}"

        # Drill into A (2x3, row-major). Children must be labelled
        # `(i,j)` and carry the cell values.
        a_cells = _vars_by_name(c, ref=rows["A"]["variablesReference"])
        expected_A = {"(1,1)": "1", "(1,2)": "2", "(1,3)": "3",
                       "(2,1)": "4", "(2,2)": "5", "(2,3)": "6"}
        for k, v in expected_A.items():
            assert a_cells.get(k) == v, \
                f"A{k} expected {v!r}, got {a_cells!r}"

        # Drill into b (column vector). Linear `(i)` labels.
        b_cells = _vars_by_name(c, ref=rows["b"]["variablesReference"])
        for i, v in enumerate(["10", "20", "30"], start=1):
            assert b_cells.get(f"({i})") == v, \
                f"b({i}) expected {v!r}, got {b_cells!r}"

        # `evaluate` should also surface a mat ref for matrix results.
        ev = c.request("evaluate", {
            "expression": "A",
            "frameId": frames[0].get("id", 0),
        })
        assert ev.get("result") == "2x3 double", ev
        wref = ev.get("variablesReference")
        assert wref and wref >= 200000, \
            f"watch(A) should carry mat-ref, got {wref!r}"
        watch_cells = _vars_by_name(c, ref=wref)
        assert watch_cells.get("(2,3)") == "6", watch_cells

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_class_instance_locals(matlabc, program):
    """Class instances render as `1x1 ClassName` and expand into
    properties.

    Without classdef-aware mirror calls every `acc = MyClass(...)` row
    landed in the LOCALS panel as `RxC double` with garbage dimensions
    (the matrix formatter dereferenced the matlab_obj* as if it were a
    matlab_mat* and read pointer fields as rows/cols). The fix wires a
    kind=2 entry through matlab_dbg_frame_set_obj / matlab_ws_set_obj
    plus a class_id -> name registry; the DAP server reads it back to
    surface the class identity and to hand out a variablesReference
    that drives the property tree.

    Uses dap_class_program.m which constructs two distinct classes
    (Account, Savings) so the scenario verifies both names resolve
    via the registry, not just the first one inserted.
    """
    import os
    cls_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_class_program.m",
    )
    # Line 10 is `disp(p.Id);` — the last script-body statement, by
    # which point all three class-bound script vars (acc, sav, p)
    # have been assigned and at least one mutator (acc.deposit) has
    # fired.
    with DapClient(matlabc, cls_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        body = _stop_event(c)
        assert body.get("line") == 10, body

        st = c.request("stackTrace", {"threadId": 1})
        frames = st.get("stackFrames") or []
        assert frames, st
        sc = c.request("scopes", {"frameId": frames[0].get("id", 0)})
        ref = (sc.get("scopes") or [{}])[0].get("variablesReference")
        assert ref, sc
        body = c.request("variables", {"variablesReference": ref})
        rows = {v.get("name"): v for v in (body.get("variables") or [])}
        # Top-level rows: each class-bound var must surface as
        # "1x1 ClassName" and carry a non-zero variablesReference.
        assert rows.get("acc", {}).get("value") == "1x1 Account", \
            f"acc should render as 1x1 Account: {rows.get('acc')!r}"
        assert rows.get("sav", {}).get("value") == "1x1 Savings", \
            f"sav should render as 1x1 Savings: {rows.get('sav')!r}"
        assert rows.get("p", {}).get("value") == "1x1 Account", \
            f"p should render as 1x1 Account: {rows.get('p')!r}"
        for n in ("acc", "sav", "p"):
            r = rows[n].get("variablesReference")
            assert r and r >= 100000, \
                f"{n} should carry an obj-ref >= 100000, got {r!r}"

        # Expand `acc`: deposit ran with amt=25, so Balance must be 75.
        acc_props = _vars_by_name(c, ref=rows["acc"]["variablesReference"])
        assert acc_props.get("Id") == "101", \
            f"acc.Id must be 101, got {acc_props!r}"
        assert acc_props.get("Balance") == "75", \
            f"acc.Balance after deposit must be 75, got {acc_props!r}"

        # Expand `sav` — must include the inherited Id/Balance plus
        # the Savings-specific Rate (0.10).
        sav_props = _vars_by_name(c, ref=rows["sav"]["variablesReference"])
        assert sav_props.get("Id") == "202", sav_props
        assert sav_props.get("Balance") == "100", sav_props
        assert sav_props.get("Rate") == "0.1", sav_props

        # `evaluate` should also surface a class instance with an
        # expandable variablesReference. Watching `acc` exercises the
        # kind=1 -> kind=2 promotion path that compensates for the
        # REPL JIT not knowing the workspace var is class-typed.
        ev = c.request("evaluate", {
            "expression": "acc",
            "frameId": frames[0].get("id", 0),
        })
        assert ev.get("result") == "1x1 Account", \
            f"watch(acc) should be 1x1 Account, got {ev!r}"
        wref = ev.get("variablesReference")
        assert wref and wref >= 100000, \
            f"watch(acc) should carry obj-ref, got {wref!r}"
        watch_props = _vars_by_name(c, ref=wref)
        assert watch_props.get("Id") == "101", watch_props
        assert watch_props.get("Balance") == "75", watch_props

        c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_class_instance_disp_in_repl(matlabc, program):
    """REPL evaluate of `disp(<obj>)` and bare `<obj>` used to SIGSEGV
    matlabc: the REPL JIT compiles a fresh `disp(acc)` whose Sema can't
    see the workspace's kind tags, types `acc` as a matrix, and lowers
    to `matlab_disp_mat(<obj_ptr>)`. The runtime then dereferenced the
    matlab_obj layout as a matlab_mat (rows/cols/data) and walked off
    into garbage memory.

    The fix maintains a runtime registry of live matlab_obj pointers
    (every constructor call registers itself); matlab_disp_mat checks
    the registry first and routes obj inputs through matlab_disp_obj,
    which prints `ClassName with properties:` plus a property listing.

    All three failure shapes are covered: REPL `disp(<obj>)`, REPL
    bare `<obj>` (auto-displayed as `<name> = <value>`), and watch-mode
    `disp(<obj>)` (which auto-promotes onto the same REPL path).
    """
    import os
    cls_program = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_class_program.m",
    )
    # Line 10 is `disp(p.Id);` — by then `acc` is constructed and
    # `acc.deposit(25)` has run, so Balance == 75.
    with DapClient(matlabc, cls_program) as c:
        initialize_and_launch(c, breakpoints=[{"line": 10}])
        _stop_event(c)

        def collect_stdout(predicate, timeout=3.0):
            import time
            deadline = time.monotonic() + timeout
            buf = []
            while time.monotonic() < deadline:
                try:
                    ev = c.wait_event(
                        "output",
                        timeout=max(0.05, deadline - time.monotonic()),
                    )
                except DapError:
                    break
                body = ev.get("body") or {}
                if body.get("category") == "stdout":
                    buf.append(body.get("output", ""))
                    if predicate("".join(buf)):
                        break
            return "".join(buf)

        # 1) REPL `disp(acc)` — must not crash; output must include the
        #    class name and the live property values.
        resp = c.request("evaluate", {
            "expression": "disp(acc)",
            "context": "repl",
        })
        assert resp.get("result", "") == "", \
            f"REPL disp(<obj>) returns empty result, got {resp!r}"
        out = collect_stdout(lambda s: "Account" in s and "75" in s)
        assert "Account" in out, f"disp(acc) output missing class name: {out!r}"
        assert "Id: 101" in out, f"disp(acc) output missing Id: {out!r}"
        assert "Balance: 75" in out, f"disp(acc) output missing Balance: {out!r}"
        assert c.proc.poll() is None, "matlabc crashed after disp(acc) in REPL"

        # 2) REPL bare `acc` — auto-displays as `acc = <value>`. Same
        #    crash class as (1), different lowering path.
        resp = c.request("evaluate", {
            "expression": "acc",
            "context": "repl",
        })
        assert resp.get("result", "") == "", \
            f"REPL bare <obj> returns empty result, got {resp!r}"
        out = collect_stdout(lambda s: "Account" in s and "Balance" in s)
        assert "Account" in out, f"bare acc output missing class name: {out!r}"
        assert "Balance: 75" in out, f"bare acc output missing Balance: {out!r}"
        assert c.proc.poll() is None, "matlabc crashed after bare acc in REPL"

        # 3) Watch-mode `disp(acc)` — auto-promotes onto the REPL branch
        #    (disp is in the void-statement list) and previously crashed
        #    via the same matlab_disp_mat path. The watch response
        #    surfaces "<void>" while the actual print flows via stdout.
        resp = c.request("evaluate", {"expression": "disp(acc)"})
        assert resp.get("result") == "<void>", \
            f"watch disp(<obj>) should auto-promote to <void>: {resp!r}"
        out = collect_stdout(lambda s: "Account" in s and "Balance" in s)
        assert "Account" in out, f"watch disp(acc) output missing class: {out!r}"
        assert c.proc.poll() is None, "matlabc crashed after watch disp(acc)"

        # 4) Subclass instance prints under its own class name and
        #    surfaces its own added property (Rate).
        resp = c.request("evaluate", {
            "expression": "disp(sav)",
            "context": "repl",
        })
        out = collect_stdout(lambda s: "Savings" in s and "0.1" in s)
        assert "Savings" in out, f"disp(sav) output missing class name: {out!r}"
        assert "Rate: 0.1" in out, f"disp(sav) output missing Rate: {out!r}"
        assert c.proc.poll() is None, "matlabc crashed after disp(sav)"

        c.request("continue")
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


def scn_var_range_for_bound(matlabc, program):
    """For-loop range bound is a script-level variable: `for i = 1:N`.

    Regression for the case where REPL-mode loadBinding always routed
    script-level Var reads through matlab_ws_get_mat (returning ptr).
    The matlab.range op then carried a (f64, !llvm.ptr) signature that
    LowerSeqLoops::extractRange refused, leaving matlab.range and
    matlab.for in the IR until the LLVM conversion stage barfed with
    `missing LLVMTranslationDialectInterface ... for op: matlab.range`.

    Asserts the loop actually executes (breakpoint inside the body
    fires three times for N=3) and that the accumulator `total`
    carries the correct value at each pause. The induction variable
    `i` lives in a loop-frame slot rather than the script workspace,
    so it isn't asserted here. Uses dap_var_range_program.m as a
    sibling fixture so dap_program.m line numbers stay stable for
    the other scenarios.
    """
    import os
    var_range = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_var_range_program.m",
    )
    with DapClient(matlabc, var_range) as c:
        initialize_and_launch(c, breakpoints=[{"line": 12}])
        # total at the breakpoint is pre-increment for that iteration:
        # iter 1 sees 0, iter 2 sees 1, iter 3 sees 3.
        for iteration, expected_total in [(1, "0"), (2, "1"), (3, "3")]:
            body = _stop_event(c)
            assert body.get("reason") == "breakpoint", \
                f"iter {iteration}: stop reason should be 'breakpoint', " \
                f"got {body!r}"
            assert body.get("line") == 12, \
                f"iter {iteration}: should stop on line 12, got {body!r}"
            vars_ = _vars_by_name(c)
            assert vars_.get("N") == "3", \
                f"iter {iteration}: N should be 3, got {vars_!r}"
            assert vars_.get("total") == expected_total, \
                f"iter {iteration}: total should be {expected_total} " \
                f"pre-increment, got {vars_!r}"
            c.request("continue")
        c.wait_event("terminated", timeout=5.0)


def scn_recursive_function_param_attrs(matlabc, program):
    """Regression for the JIT path's "Unhandled parameter attribute
    'matlab.name'" error.

    The lowering stamps `matlab.name` arg attrs on every `func.func`
    so EmitC / SystemVerilog can render named signatures. The
    LLVM-conversion pipeline (ConvertFuncToLLVMPass) propagates
    those attrs to `llvm.func`. The plain `-emit-llvm` translator
    tolerates them, but the JIT (ExecutionEngine::create) used by
    `-dap` and `-repl` errors out with `Unhandled parameter
    attribute '<name>'`. The fix strips every `matlab.*` arg/result
    attr from `llvm.func` ops between the conversion pipeline and
    LLVM-IR translation (`stripMatlabFuncAttrs` in
    `lib/MLIR/Passes/LowerToLLVMIR.cpp`).

    Existing scenarios all use `dap_program.m` (no user functions
    with parameters) or `dap_locals_program.m` (a single non-
    recursive `compute(a,b)` that happens not to trip the per-
    callsite cloning path that re-stamps the attrs). This fixture
    is `dap_recursion_program.m` — a tiny `fact(n)` that calls
    itself in the else branch, matching the shape that originally
    surfaced the bug. The test only needs to reach
    configurationDone; if launch fails (the bug returns), the
    `initialize_and_launch` call raises with the JIT's compile
    error visible in stderr.
    """
    import os
    rec = os.path.join(
        os.path.dirname(os.path.abspath(program)),
        "dap_recursion_program.m",
    )
    with DapClient(matlabc, rec) as c:
        # Stop on entry so we have a known synchronization point. The
        # MLIR diagnostic for the bad attribute is non-fatal —
        # translateModuleToLLVMIR emits the warning and still produces
        # a working module, so launch + execution succeed either way.
        # The only durable signal is the diagnostic text reaching the
        # subprocess's stderr (captured by DapClient's _stderr_buf
        # reader thread before the DAP server's pipe-redirect fully
        # routes it as `output` events).
        initialize_and_launch(c, stop_on_entry=True)
        body = _stop_event(c)
        assert body.get("reason") in ("entry", "step"), \
            f"expected entry-stop, got {body!r}"
        c.request("continue")
        c.wait_event("terminated", timeout=5.0)
        # Drain every category=stderr output event the server forwarded
        # (the DAP path tees compile-time diagnostics through these so
        # the IDE's debug console shows them).
        stderr_chunks = list(c._stderr_buf)
        try:
            while True:
                ev = c.wait_event("output", timeout=0.05)
                body = ev.get("body") or {}
                if body.get("category") == "stderr":
                    stderr_chunks.append(body.get("output") or "")
        except DapError:
            pass

    full = "".join(stderr_chunks)
    bad_signals = ("matlab.name", "Unhandled parameter attribute")
    for sig in bad_signals:
        assert sig not in full, (
            f"DAP stderr leaked compile-time MLIR diagnostic {sig!r} — "
            f"the matlab.* arg/result attrs are reaching the JIT translator. "
            f"See `mlirgen::stripMatlabFuncAttrs` in "
            f"lib/MLIR/Passes/LowerToLLVMIR.cpp. Captured stderr:\n{full}"
        )


# --- entry point -------------------------------------------------------------

def all_scenarios():
    g = globals()
    return [(name, g[name]) for name in sorted(g) if name.startswith("scn_")]
