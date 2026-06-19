#!/usr/bin/env python3
"""Bare-minimum CPU-lane gate for examples/gpu (no GPU required).

GitHub's runners have no GPU, and the examples sweep (run_sweep.sh) is
AOT-only and baselines every GPU example as a known failure — so the
JIT / -dap / -repl compile path of examples/gpu had no per-PR coverage.
That's exactly where three bugs hid:

  * rand(n, n, 'single') dtype-string lowering          (#319)
  * sibling .m scan tanking the whole directory's compile (#320)
  * none-typed param feeding gpuArray.* not promoted in  (#326)
    the JIT pipeline

The gpuArray CPU-debug lane is host-BLAS identity, so all three are
plain CPU compile/lowering checks. This test pins them on the *real*
example files:

  1. `-emit-llvm` must lower the in-scope files cleanly (guards #319 and
     any future lowering regression in them).
  2. `-dap` (JIT) must accept `launch` (i.e. compileProgram succeeds) for
     a representative file *in the real examples/gpu directory* — which
     contains the advanced stencil/parfor siblings. This guards #320
     (the sibling scan must not pull them in and abort) and #326 (the
     param `n` feeding gpuArray.rand must promote). We assert only the
     compile (launch accepted), not a clean run: a function-file entry
     can't run to a clean exit (param unbound), and -dap has a known
     teardown SIGSEGV race orthogonal to what we're guarding.

Usage: gpu_cpu_lane.py <path-to-matlabc>
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

    print("examples/gpu CPU-lane gate:")
    print("  -emit-llvm (lowering):")
    for name in EXPECT_LOWER:
        ok, err = emit_lowers(matlabc, os.path.join(GPU_DIR, name))
        print(f"    {name:30s} {'ok' if ok else 'FAIL: ' + err[:60]}")
        if not ok:
            failures.append(f"emit-llvm {name}: {err[:80]}")

    print("  -dap (JIT compile):")
    for name in EXPECT_JIT_COMPILE:
        ok, err = jit_compiles(matlabc, os.path.join(GPU_DIR, name))
        print(f"    {name:30s} {'ok' if ok else 'FAIL: ' + err}")
        if not ok:
            failures.append(f"jit-compile {name}: {err}")

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
