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
    ]

    failed = []
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

    print("----")
    print(f"passed: {len(cases) - len(failed)}    failed: {len(failed)}")
    for name, msg in failed:
        print(f"\n=== {name} ===\n  {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
