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
        # dropping the connection.
        try:
            c.request("evaluate", {"expression": "1 +"})
            raise AssertionError("evaluate accepted a malformed expression")
        except DapError:
            pass

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
