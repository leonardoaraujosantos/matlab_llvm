#!/usr/bin/env python3
"""Per-PR CPU-lane gate for examples/ (no GPU required).

The examples sweep (run_sweep.sh) is AOT-only — it never exercises the
JIT / -dap / -repl compile path, which can reject a program the AOT
`-emit-llvm` path accepts (different pass pipeline / ReplMode / slot-
promotion ordering). That path had no per-PR coverage, and three bugs hid
there in the GPU examples alone:

  * rand(n, n, 'single') dtype-string lowering          (#319)
  * sibling .m scan tanking the whole directory's compile (#320)
  * none-typed param feeding gpuArray.* not promoted in  (#326)
    the JIT pipeline

The gpuArray CPU-debug lane is host-BLAS identity, so these are plain CPU
checks. This test pins:

  1. `-emit-llvm` lowering of the in-scope GPU example files (guards #319
     and any future lowering regression in them).
  2. `-dap` (JIT) `launch` acceptance (compileProgram OK) for files in the
     real examples/gpu directory — which holds the advanced stencil/parfor
     siblings — guarding #320 (sibling scan must not abort the compile) and
     #326 (the param feeding gpuArray.rand must promote).
  3. `-dap` (JIT) `launch` acceptance for a representative .m from each
     major toolbox directory, so an AOT-OK-but-JIT-broken regression in any
     toolbox (the #326 class, but non-GPU) is caught per-PR.

We assert only the compile (launch accepted), not a clean run: a function-
file entry can't run to a clean exit (param unbound), and -dap has a known
teardown SIGSEGV race orthogonal to what we're guarding.

Usage: examples_cpu_lane.py <path-to-matlabc>
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
GPU_DIR = os.path.join(REPO, "examples", "gpu")

# Files whose `-emit-llvm` must lower cleanly on the CPU lane today.
# (stencil2d / parfor_* / run_validation_suite are documented carve-outs:
# zeros(...,'like',A), sum(x,'all'), norm(x,'fro'), multi-file resolution.)
EXPECT_LOWER = [
    "benchmark_gpu_backend.m",
    "mandelbrot_gpu.m",
    "test_gpuarray_arrayfun.m",
    "test_gpuarray_axpy.m",
    "test_gpuarray_gemm.m",
]

# Files that must JIT-compile under -dap (launch accepted) — driven from
# the real examples/gpu directory so the sibling scan and param promotion
# are exercised exactly as the IDE hits them.
EXPECT_JIT_COMPILE = [
    "test_gpuarray_gemm.m",   # param `n` -> gpuArray.rand(n,n,'single') (#326)
    "mandelbrot_gpu.m",       # simple file in a dir of advanced siblings (#320)
]

# One representative script per major toolbox directory (paths relative to
# the repo root). Each must JIT-compile under -dap, so an AOT-OK-but-JIT-
# broken regression in any toolbox is caught — the #326 class generalised
# beyond GPU. All verified to JIT-compile when added; if one legitimately
# stops working, fix the JIT path or swap the representative (don't silently
# drop coverage for that toolbox).
TOOLBOX_JIT_COMPILE = [
    "examples/matrix_mult.m",
    "examples/eigendecomp.m",
    "examples/solve_linear.m",
    "examples/ode_solver.m",
    "examples/control/balreal_demo.m",
    "examples/signal/bandpass_design.m",
    "examples/stats_ml/distribution_fitting.m",
    "examples/optim/blade_pitch_opt.m",
    "examples/pde/antenna_glb_fem.m",
    "examples/comm/alamouti_diversity.m",
    "examples/dsp/adaptive_eq.m",
]


def emit_lowers(matlabc: str, path: str) -> tuple[bool, str]:
    p = subprocess.run([matlabc, "-emit-llvm", path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       env={**os.environ, "LC_ALL": "C"})
    err = p.stderr.decode(errors="replace")
    bad = [l for l in err.splitlines() if "error:" in l.lower()]
    return (p.returncode == 0 and not bad), (bad[0] if bad else "")


def jit_compiles(matlabc: str, path: str) -> tuple[bool, str]:
    """True iff the -dap server accepts `launch` (compileProgram OK)."""
    sys.path.insert(0, os.path.join(REPO, "test", "Debug"))
    from dap_client import DapClient, initialize_and_launch  # noqa
    try:
        with DapClient(matlabc, path) as c:
            initialize_and_launch(c, stop_on_entry=True)  # raises on compile fail
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def main(matlabc: str) -> int:
    matlabc = os.path.abspath(matlabc)
    failures: list[str] = []

    print("examples CPU-lane gate:")
    print("  gpu -emit-llvm (lowering):")
    for name in EXPECT_LOWER:
        ok, err = emit_lowers(matlabc, os.path.join(GPU_DIR, name))
        print(f"    {name:38s} {'ok' if ok else 'FAIL: ' + err[:60]}")
        if not ok:
            failures.append(f"emit-llvm {name}: {err[:80]}")

    print("  gpu -dap (JIT compile):")
    for name in EXPECT_JIT_COMPILE:
        ok, err = jit_compiles(matlabc, os.path.join(GPU_DIR, name))
        print(f"    {name:38s} {'ok' if ok else 'FAIL: ' + err}")
        if not ok:
            failures.append(f"jit-compile {name}: {err}")

    print("  toolbox -dap (JIT compile):")
    for rel in TOOLBOX_JIT_COMPILE:
        ok, err = jit_compiles(matlabc, os.path.join(REPO, rel))
        print(f"    {rel:38s} {'ok' if ok else 'FAIL: ' + err}")
        if not ok:
            failures.append(f"jit-compile {rel}: {err}")

    print("----")
    if failures:
        print(f"FAIL  {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <matlabc>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
