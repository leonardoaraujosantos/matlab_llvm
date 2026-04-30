#!/usr/bin/env python3
"""Phase 6 DAP test for the .mflow file-type hook.

Verifies that `matlabc -dap` accepts a `.mflow` program and honours
breakpoints set on the original `.mflow` JSON line numbers — i.e.
that the GraphToAST source-range remap (every synthesized statement
carries its originating block's `.mflow` byte offset as Range.Begin)
makes its way to the runtime debug-hook table and the DAP frame
emission.

The test reuses the existing DapClient harness from test/Debug/.
See docs/flowchart_frontend.md and docs/debug.md for the protocol
surface.
"""

import os
import sys


def find_block_line(mflow_path, markers):
    """Return the 1-based line number of the first line containing
    any of `markers` in the .mflow file. The list lets the test
    tolerate both compact and IDE-pretty-printed JSON spacing."""
    if isinstance(markers, str):
        markers = [markers]
    with open(mflow_path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            for m in markers:
                if m in line:
                    return i
    raise RuntimeError(f"none of markers {markers!r} found in {mflow_path}")


def _stop_event(client, timeout=5.0):
    ev = client.wait_event("stopped", timeout=timeout)
    return ev.get("body") or {}


def main():
    if len(sys.argv) != 2:
        print("usage: run_tests.py <path-to-matlabc>", file=sys.stderr)
        return 2
    matlabc = os.path.realpath(sys.argv[1])

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    debug_harness = os.path.join(repo_root, "test", "Debug")
    sys.path.insert(0, debug_harness)
    from dap_client import DapClient, DapError, initialize_and_launch  # noqa: E402

    # The Phase 6 contract: a breakpoint set on a `.mflow` JSON line
    # verifies, fires with reason="breakpoint", and the top stack
    # frame's source.path is the .mflow file. We don't assert the
    # post-resume stdout — output forwarding through DAP `output`
    # events is exercised by the existing debug-dap-tests lane and
    # is independent of which frontend produced the TU.
    # Marker substrings tolerate either compact (hand-written) or
    # IDE-pretty-printed (`" : "` around the colon) JSON formats so
    # the test stays robust regardless of whether the fixture was
    # last saved by the IDE.
    cases = [
        # for_loop.mflow: bp on the `init` variable block (`name: total`).
        # When hit, execution should pause before the for loop runs.
        {
            "name": "for_loop_bp_on_init",
            "program": os.path.join(repo_root, "examples/mflow/for_loop.mflow"),
            "markers": ['"name" : "total"', '"name": "total"'],
        },
        # hello.mflow: bp on the display block.
        {
            "name": "hello_bp_on_display",
            "program": os.path.join(repo_root, "examples/mflow/hello.mflow"),
            "markers": ['"expression" : "\'Hello, world!\'"',
                        '"expression": "\'Hello, world!\'"'],
        },
        # nested_for_if (control-flow block): bp on the inner `if`
        # block fires inside the for loop body.
        {
            "name": "nested_if_bp",
            "program": os.path.join(repo_root,
                                    "test/Flowchart/EmitMatlab/nested_for_if.mflow"),
            "markers": ['"kind" : "if"', '"kind": "if"'],
        },
        # even_odd.mflow: for-loop iter is "1:N" where N is a
        # script-level variable. Regression for the case where REPL
        # mode boxed N through matlab_ws_get_mat (returning ptr) and
        # the resulting matlab.range with a !llvm.ptr operand survived
        # all lowering passes, surfacing as
        # `missing LLVMTranslationDialectInterface for op: matlab.range`
        # at JIT time. A breakpoint on the body's if block proves the
        # loop actually executed (i.e. matlab.range/matlab.for were
        # lowered cleanly). The find_block_line helper picks the FIRST
        # `mod(i, 2) == 0` occurrence, which is the if-block's `cond`
        # field — the inside-the-loop breakpoint.
        {
            "name": "var_range_bp_in_loop_body",
            "program": os.path.join(repo_root, "examples/mflow/even_odd.mflow"),
            "markers": ['"cond" : "mod(i, 2) == 0"',
                        '"cond": "mod(i, 2) == 0"'],
        },
    ]

    failed = []

    # Phase 8b: step-over collapses a multi-statement `expression`
    # block to a single logical step. The fixture's `multi` block
    # has `data.expression = "v = v + 1; w = v * 2; z = w - 3"`
    # (three Stmts). Stop on `v0`, issue `next` twice, assert the
    # second `next` lands on `show` (skipping all three multi-stmts
    # as one block-step).
    sys.stdout.write("  step_over_multi_stmt_block ... ")
    sys.stdout.flush()
    try:
        prog = os.path.join(repo_root, "test/Flowchart/Dap/multi_stmt.mflow")
        with open(prog) as f:
            blob = f.read().splitlines()
        v0_line   = next(i+1 for i, l in enumerate(blob) if '"id": "v0"' in l)
        show_line = next(i+1 for i, l in enumerate(blob) if '"id": "show"' in l)
        with DapClient(matlabc, prog) as c:
            initialize_and_launch(c, breakpoints=[{"line": v0_line}])
            _stop_event(c)  # stopped at v0
            c.request("next")
            _stop_event(c)  # stopped at multi
            c.request("next")
            body = _stop_event(c)  # should be at show, NOT inside multi
            actual = body.get("line")
            if actual != show_line:
                raise DapError(
                    f"expected next() to skip past multi block to show "
                    f"(line {show_line}), got line {actual}")
            c.request("continue")
            c.wait_event("terminated", timeout=10.0)
        print("ok")
    except Exception as e:
        print("FAIL")
        failed.append(("step_over_multi_stmt_block", str(e)))

    # Regression: a breakpoint set on a JSON line INSIDE a `display`
    # block (e.g. the `data.expression` line) must resolve to that
    # block, not snap forward to the next one. Pre-fix only the line
    # of the opening `{` was registered for each block, so a click on
    # any other line of the block was forward-snapped to the next
    # block's `{` — making it look as if `display` blocks couldn't
    # be broken on at all. The fix tags every Stmt's range with the
    # closing `}` and aliases interior lines back to the begin line.
    sys.stdout.write("  display_bp_on_inner_line ... ")
    sys.stdout.flush()
    try:
        prog = os.path.join(repo_root, "examples/mflow/hello.mflow")
        with open(prog) as f:
            blob = f.read().splitlines()
        # Line of the display block's expression field — NOT the
        # block's opening `{`. Tolerates compact + IDE-pretty JSON.
        markers = ['"expression" : "\'Hello, world!\'"',
                   '"expression": "\'Hello, world!\'"']
        expr_line = None
        for i, l in enumerate(blob, start=1):
            if any(m in l for m in markers):
                expr_line = i
                break
        if expr_line is None:
            raise DapError(f"expression line not found in {prog}")
        # Sanity: the block's `{` is on the line above, so the bp
        # we're about to set is genuinely on a non-canonical row.
        d1_open = next(i+1 for i, l in enumerate(blob)
                       if '"id": "d1"' in l or '"id" : "d1"' in l)
        if expr_line == d1_open:
            raise DapError("test fixture changed: expression and `{` "
                           f"now share line {expr_line}; pick a "
                           "different fixture")
        with DapClient(matlabc, prog) as c:
            initialize_and_launch(
                c, stop_on_entry=False,
                breakpoints=[{"line": expr_line}])
            body = _stop_event(c)
            if body.get("reason") != "breakpoint":
                raise DapError(f"expected reason=breakpoint, got {body!r}")
            st = c.request("stackTrace", {"threadId": 1})
            frames = st.get("stackFrames") or []
            top_name = (frames[0] if frames else {}).get("name") or ""
            if "[block:d1]" not in top_name:
                raise DapError(
                    f"expected break on display block d1, top frame "
                    f"name was {top_name!r} (likely snapped forward "
                    "to the next block — alias rewrite is broken)")
            c.request("continue")
            c.wait_event("terminated", timeout=10.0)
        print("ok")
    except Exception as e:
        print("FAIL")
        failed.append(("display_bp_on_inner_line", str(e)))

    for case in cases:
        sys.stdout.write(f"  {case['name']} ... ")
        sys.stdout.flush()
        try:
            line = find_block_line(case["program"], case["markers"])
            with DapClient(matlabc, case["program"]) as c:
                initialize_and_launch(
                    c, stop_on_entry=False,
                    breakpoints=[{"line": line}])

                body = _stop_event(c)
                if body.get("reason") != "breakpoint":
                    raise DapError(
                        f"expected reason=breakpoint, got {body!r}")

                # Walk the stack: the top frame should point at the
                # .mflow file, with a line at-or-near the breakpoint
                # (the runtime may snap to the next executable line
                # within the block). Phase 8a: the frame name carries
                # the originating block id so the IDE can highlight
                # the active block on the canvas.
                st = c.request("stackTrace", {"threadId": 1})
                frames = st.get("stackFrames") or []
                if not frames:
                    raise DapError("empty stackTrace at breakpoint")
                top_name = frames[0].get("name") or ""
                if "[block:" not in top_name:
                    raise DapError(
                        f"top frame name missing block-id tag: "
                        f"{top_name!r}")
                src_path = (frames[0].get("source") or {}).get("path") or ""
                if not src_path.endswith(".mflow"):
                    raise DapError(
                        f"top frame source not a .mflow: {src_path!r}")

                # Resume and let the program run to termination. We
                # don't assert on stdout — the DAP output-forwarding
                # surface is exercised by debug-dap-tests; this lane
                # is about the .mflow → DAP plumbing.
                # Some breakpoints fire multiple times (e.g. a bp on
                # an `if` block inside a for loop); resume past each
                # stop until termination.
                import time as _time
                deadline = _time.monotonic() + 10
                c.request("continue")
                while _time.monotonic() < deadline:
                    try:
                        ev = c.wait_event("stopped", timeout=0.5)
                        if (ev.get("body") or {}).get("reason") == "breakpoint":
                            c.request("continue")
                    except DapError:
                        # No more stops; check termination.
                        try:
                            c.wait_event("terminated", timeout=0.5)
                            break
                        except DapError:
                            continue
                else:
                    raise DapError("did not terminate within 10s")
            print("ok")
        except Exception as e:
            print("FAIL")
            failed.append((case["name"], str(e)))

    # +1 for the standalone Phase 8b case driven above.
    total = len(cases) + 1
    print("----")
    print(f"passed: {total - len(failed)}    failed: {len(failed)}")
    for name, msg in failed:
        print(f"\n=== {name} ===\n  {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
