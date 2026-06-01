#!/usr/bin/env python3
"""Regression: matlabc -dap (JIT) parity with the static -emit-* paths.

Background — the JIT used to bail out of `launch` with an empty
"failed to compile program" response whenever a user function returned
or mirrored a logical / boolean value (`r = age > 18` etc.). The root
cause was twofold:

  1. The JIT pipeline (REPL + DAP) skipped `runRefineFuncSigs`, which
     the static -emit-* paths run to patch up function signatures
     after `LowerScalarsToArith` rewrites a return into `i1`. Without
     it the function body produced `i1` while the signature still
     declared `-> none`, and the verifier / func-to-llvm conversion
     refused the module.
  2. `LowerTensorOps`'s `matlab_dbg_frame_set` and `matlab_ws_set_*`
     rewrites only handled f64 / ptr value types. An i1 mirror call
     emitted by emitStore (DebugMode is always on in the JIT path)
     survived all subsequent passes; the conversion then failed with
     `cannot be converted to LLVM IR: ... matlab.const_char`.

The fix runs `runRefineFuncSigs` after the early scalar fold and the
late slot-lowering pass, and teaches both rewrites to cast integer
mirror values to f64 (i1 via `select(v, 1.0, 0.0)`, wider ints via
sitofp/uitofp). Plus the DAP server now spawns the stderr-pipe
reader BEFORE accepting `launch` so MLIR diagnostics surface as
`output` events instead of getting buffered until configurationDone,
and `sharedDapContext` registers an MLIR diagnostic handler so a
future verifier failure can't go silent again.

This test guards the parity:

  * For each program shape, we DAP-launch and assert the JIT got past
    compileProgram (i.e. no "failed to compile program").
  * For each example in `examples/mflow/` we also assert the program
    runs to termination through the DAP `continue` path.

Programs are written into per-test isolated directories so that the
DAP path's auto-load-siblings logic doesn't pull in random `.m` files
from /tmp.
"""

import os
import shutil
import sys
import time


def main():
    if len(sys.argv) != 2:
        print("usage: run_jit_userfn_tests.py <path-to-matlabc>",
              file=sys.stderr)
        return 2
    matlabc = sys.argv[1]
    if not os.path.isfile(matlabc) or not os.access(matlabc, os.X_OK):
        print(f"matlabc not executable: {matlabc}", file=sys.stderr)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from dap_client import DapClient, DapError, initialize_and_launch  # noqa

    repo = os.path.normpath(os.path.join(here, "..", ".."))
    mflow_dir = os.path.join(repo, "examples/mflow")

    # User-function shape matrix. Every case must reach the post-launch
    # state — that's the JIT-compiled-OK signal — except for cases
    # marked `static_fails=True`, which the static -emit-* path
    # itself can't lower today (matlab.matmul (i1,f64), matlab.short_and).
    # We assert THOSE fail at JIT in the same spot so the parity gap
    # stays visible; flipping them to OK becomes a follow-up signal.
    cases = [
        # name, source, static_fails
        ("scalar_double_no_args",
         "disp(get_pi());\n"
         "function y = get_pi()\n  y = 3.14;\nend\n",
         False),
        ("scalar_double_one_arg",
         "disp(squarem(4));\n"
         "function y = squarem(x)\n  y = x * x;\nend\n",
         False),
        ("scalar_double_two_args",
         "disp(add(2, 3));\n"
         "function y = add(a, b)\n  y = a + b;\nend\n",
         False),
        # The original is_old.m shape — bool return.
        ("logical_return_gt_const",
         "disp(over18(25));\n"
         "function r = over18(age)\n  r = age > 18;\nend\n",
         False),
        # Bool from `==`.
        ("logical_via_eq",
         "disp(eq5(5));\n"
         "function r = eq5(x)\n  r = (x == 5);\nend\n",
         False),
        ("void_function",
         "say_hi();\n"
         "function say_hi()\n  disp('hi');\nend\n",
         False),
        ("recursive_factorial",
         "disp(fact(5));\n"
         "function y = fact(n)\n  if n <= 1\n    y = 1;\n"
         "  else\n    y = n * fact(n - 1);\n  end\nend\n",
         False),
        ("if_inside_no_recursion",
         "disp(clamp_pos(7));\n"
         "function y = clamp_pos(x)\n  if x > 0\n    y = x;\n"
         "  else\n    y = 0;\n  end\nend\n",
         False),
        ("two_returns",
         "[a, b] = pair();\ndisp(a); disp(b);\n"
         "function [u, v] = pair()\n  u = 1; v = 2;\nend\n",
         False),
        ("nested_user_call",
         "disp(outer(3));\n"
         "function y = outer(x)\n  y = inner(x) + 1;\nend\n"
         "function z = inner(x)\n  z = x * 2;\nend\n",
         False),
        # Script-scope bool variable — exercises matlab_ws_set_* on
        # an `i1` operand. Different code path from the function-frame
        # mirror (matlab_dbg_frame_set) the cases above hit; the
        # workspace store fires from the script body via the ReplMode
        # branch in Lowering.cpp.
        ("script_scope_bool_var",
         "x = 25 > 18;\ndisp(x);\n",
         False),
        # Function handles (issue #77). A workspace-backed handle (`f`
        # isn't a local slot in the JIT path) round-trips through
        # matlab_ws_set/get_handle (kind=13). `f(0)` must lower through
        # the matlab_call_handle_s* trampoline (builtin @sin) and `p(6)`
        # through a direct call to the user function (@sq) — NOT a matrix
        # subscript on the code pointer, which used to crash. The whole
        # program compiles as one module here, so @sq is in scope.
        ("function_handle_builtin_and_user",
         "f = @sin;\ndisp(f(0));\n"
         "p = @sq;\ndisp(p(6));\n"
         "function y = sq(x)\n  y = x * x;\nend\n",
         False),
        # Struct-returning runtime call that yields NULL (issue #77).
        # pde_load_glb on a missing file returns a NULL matlab_struct*.
        # In the JIT/REPL path `s` is workspace-backed, so it round-trips
        # through the script-scope store/load. It must store via
        # matlab_ws_set_struct (kind=12) and read back the NULL faithfully
        # — NOT as a fresh mat_alloc(0,0) that pde_mesh_nodes then
        # dereferences as a struct (struct_find_field walking a mat layout
        # → heap-dependent SIGSEGV). With the fix the program degrades
        # gracefully: pde_mesh_nodes(NULL) → empty, size → 0, terminates.
        ("struct_returning_null_roundtrip",
         "s = pde_load_glb('/tmp/__no_such_model_77__.glb');\n"
         "n = pde_mesh_nodes(s);\n"
         "disp(size(n, 1));\n"
         "disp('ok');\n",
         False),
    ]

    work = os.path.join("/tmp", "matlabc_jit_userfn_test")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    failed = []

    def jit_compiles(prog_path, allow_compile_fail=False):
        """Returns (ok_compile, error_text). ok_compile is True iff
        the DAP server accepted `launch`. error_text is whatever
        showed up on the stderr-output channel (used to fail loudly
        on regressions)."""
        try:
            with DapClient(matlabc, prog_path) as c:
                err = []
                t0 = time.monotonic()
                try:
                    initialize_and_launch(c, stop_on_entry=True)
                except DapError as e:
                    # Drain output briefly to capture the diagnostic.
                    deadline = time.monotonic() + 0.7
                    while time.monotonic() < deadline:
                        try:
                            ev = c.wait_event("output", timeout=0.2)
                            b = ev.get("body") or {}
                            if b.get("category") == "stderr":
                                err.append(b.get("output", ""))
                        except DapError:
                            pass
                    return False, "".join(err) or str(e)
                # Compile succeeded — drive to termination so the
                # JIT'd worker exits cleanly. Some programs print
                # to stdout; we don't assert content, only that the
                # `terminated` event arrives.
                try:
                    c.wait_event("stopped", timeout=5.0)
                except DapError:
                    pass
                c.request("continue")
                deadline = time.monotonic() + 8.0
                terminated = False
                while time.monotonic() < deadline:
                    try:
                        ev = c.wait_event("terminated", timeout=0.4)
                        terminated = True
                        break
                    except DapError:
                        # Maybe a follow-on `stopped` (multi-bp); resume.
                        try:
                            c.wait_event("stopped", timeout=0.1)
                            c.request("continue")
                        except DapError:
                            pass
                if not terminated:
                    return False, f"did not terminate within {time.monotonic()-t0:.1f}s"
                return True, ""
        except Exception as e:
            return False, str(e)

    print(f"User-function JIT shape matrix ({len(cases)} cases):")
    for name, src, static_fails in cases:
        d = os.path.join(work, name)
        os.makedirs(d, exist_ok=True)
        prog = os.path.join(d, "prog.m")
        with open(prog, "w") as f:
            f.write(src)
        sys.stdout.write(f"  {name:<32} ... ")
        sys.stdout.flush()
        ok, err = jit_compiles(prog)
        if ok:
            print("ok")
        else:
            print("FAIL")
            failed.append((name, err))

    # examples/mflow/ — every example must JIT + run.
    print(f"\nexamples/mflow/ end-to-end JIT (every .mflow):")
    mflows = sorted(f for f in os.listdir(mflow_dir) if f.endswith(".mflow"))
    for fname in mflows:
        prog = os.path.join(mflow_dir, fname)
        sys.stdout.write(f"  {fname:<32} ... ")
        sys.stdout.flush()
        ok, err = jit_compiles(prog)
        if ok:
            print("ok")
        else:
            print("FAIL")
            failed.append((f"mflow:{fname}", err))

    # REPL pipeline parity. The REPL JIT (`matlabc -repl`, via
    # `runReplInput`) shares the same MLIR pass pipeline as the DAP
    # path; without the `runRefineFuncSigs` insertion AND the
    # `matlab_ws_set_*` int-operand handling, a script-scope bool
    # store ("x = 25 > 18; disp(x);") would print
    #   error: ExecutionEngine::create failed: could not convert to LLVM IR
    # because the matlab_ws_set_f64 call would carry an i1 operand
    # the conversion can't lower. We feed the same input via stdin
    # and assert no `error:` line appears.
    #
    # Note: a *standalone* function definition at REPL with no caller
    # in scope (e.g. just `function r = over18(age); r = age > 18; end`)
    # has a separate bug — args stay `none`-typed, the function body's
    # matlab.gt(none, f64) survives translation. The static
    # `-emit-llvm` path fails the same way on function-only input.
    # That gap is broader than this fix and isn't covered here.
    print(f"\nREPL parity (script-scope bool):")
    sys.stdout.write(f"  repl_script_scope_bool           ... ")
    sys.stdout.flush()
    import subprocess
    repl_in = b"x = 25 > 18; disp(x);\nexit\n"
    try:
        proc = subprocess.run([matlabc, "-repl"], input=repl_in,
                              capture_output=True, timeout=15)
        combined = (proc.stdout + b"\n" + proc.stderr).decode(
            "utf-8", "replace")
        if "error:" in combined.lower():
            print("FAIL")
            failed.append(("repl_script_scope_bool",
                           f"REPL output contains 'error:':\n{combined}"))
        else:
            print("ok")
    except subprocess.TimeoutExpired:
        print("FAIL")
        failed.append(("repl_script_scope_bool", "REPL timed out"))
    except Exception as e:
        print("FAIL")
        failed.append(("repl_script_scope_bool", f"REPL spawn failed: {e}"))

    # REPL function-handle parity (issue #77). A named handle assigned
    # on one turn (`f = @sqrt;`) and called on a later turn
    # (`disp(f(16));`) used to crash: `f` round-tripped through the
    # workspace as a kind=1 matrix, so the call lowered to
    # matlab_subscript1_s on the stored code pointer (SIGSEGV / wrong
    # `0`). With the kind=13 handle ABI + the matlab_call_handle_s*
    # trampoline the call invokes the pointer directly. We assert the
    # process exits cleanly (no 139/SIGSEGV) and prints the right value.
    print(f"\nREPL parity (cross-turn function handle):")
    sys.stdout.write(f"  repl_handle_cross_turn           ... ")
    sys.stdout.flush()
    handle_in = b"f = @sqrt;\ndisp(f(16));\nexit\n"
    try:
        proc = subprocess.run([matlabc, "-repl"], input=handle_in,
                              capture_output=True, timeout=15)
        combined = (proc.stdout + b"\n" + proc.stderr).decode(
            "utf-8", "replace")
        if proc.returncode not in (0,):
            print("FAIL")
            failed.append(("repl_handle_cross_turn",
                           f"non-zero exit {proc.returncode} "
                           f"(139 = SIGSEGV):\n{combined}"))
        elif "error:" in combined.lower():
            print("FAIL")
            failed.append(("repl_handle_cross_turn",
                           f"REPL output contains 'error:':\n{combined}"))
        elif "\n4\n" not in ("\n" + combined + "\n"):
            print("FAIL")
            failed.append(("repl_handle_cross_turn",
                           f"expected sqrt(16)=4 in output:\n{combined}"))
        else:
            print("ok")
    except subprocess.TimeoutExpired:
        print("FAIL")
        failed.append(("repl_handle_cross_turn", "REPL timed out"))
    except Exception as e:
        print("FAIL")
        failed.append(("repl_handle_cross_turn", f"REPL spawn failed: {e}"))

    # Compile-error visibility: the DAP server used to spawn its
    # stderr-pipe reader at `configurationDone`, so an MLIR diagnostic
    # raised during `launch` got buffered with no reader, and the
    # `failed to compile program` response arrived with an empty
    # stderr. The fix spawns the reader before any DAP frame is
    # processed AND attaches an MLIR diagnostic handler to
    # sharedDapContext. Both must hold for diagnostics to surface;
    # this case guards both by deliberately tripping the LLVM-conversion
    # pipeline (matlab.short_and isn't lowered today) and asserting
    # we receive a non-empty `output` event with category="stderr"
    # carrying the actual diagnostic text.
    print(f"\nCompile-error visibility:")
    sys.stdout.write(f"  diag_visible_on_failed_launch    ... ")
    sys.stdout.flush()
    broken_dir = os.path.join(work, "broken_undef_field")
    os.makedirs(broken_dir, exist_ok=True)
    broken = os.path.join(broken_dir, "prog.m")
    # Field-access on a numeric literal lowers to `matlab.undef`,
    # which the LLVM translation lane rejects with a clean
    # diagnostic carrying the "matlab.undef" op name.
    #
    # The historical fixture used `(a > 0) && (b > 0)` to exercise
    # the same path via an unsupported short-circuit boolean, but
    # LowerScalarsToArith now lowers short_and / short_or cleanly
    # — that program compiles and launches fine. Field-access on a
    # number stays a launch-time error.
    with open(broken, "w") as f:
        f.write("x = 1.bad_field;\ndisp(x);\n")
    try:
        with DapClient(matlabc, broken) as c:
            c.request("initialize", {"linesStartAt1": True,
                                     "columnsStartAt1": True})
            c.wait_event("initialized", timeout=5.0)
            launch_failed = False
            try:
                c.request("launch", {"program": broken, "stopOnEntry": True})
            except DapError:
                launch_failed = True
            err_chunks = []
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                try:
                    ev = c.wait_event("output", timeout=0.3)
                    b = ev.get("body") or {}
                    if b.get("category") == "stderr":
                        err_chunks.append(b.get("output", ""))
                except DapError:
                    pass
            err_text = "".join(err_chunks).strip()
            if not launch_failed:
                print("FAIL")
                failed.append(("diag_visible_on_failed_launch",
                               "expected launch to fail; it succeeded"))
            elif not err_text:
                print("FAIL")
                failed.append(("diag_visible_on_failed_launch",
                               "launch failed but no stderr output event "
                               "arrived (early reader / diag handler "
                               "regression)"))
            elif "matlabc" not in err_text and "matlab." not in err_text:
                print("FAIL")
                failed.append(("diag_visible_on_failed_launch",
                               f"stderr output didn't carry an MLIR "
                               f"diagnostic; got: {err_text[:200]!r}"))
            else:
                print("ok")
    except Exception as e:
        print("FAIL")
        failed.append(("diag_visible_on_failed_launch", str(e)))

    total = len(cases) + len(mflows) + 3  # +2 REPL, +1 visibility
    print("----")
    print(f"passed: {total - len(failed)}    "
          f"failed: {len(failed)}")
    for name, msg in failed:
        sep = "\n    "
        msg_short = sep.join(msg.strip().splitlines()[:6]) or "(no detail)"
        print(f"\n=== {name} ===\n    {msg_short}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
