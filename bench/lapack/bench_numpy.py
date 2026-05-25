#!/usr/bin/env python3
"""NumPy companion benchmarks — one Python entry point, kernel + N from env.

Driven by BENCH_KERNEL and BENCH_N. Keeping all NumPy variants in one file
shaves NumPy import overhead per trial (a few hundred ms each cold start
otherwise dominates the smaller sizes).
"""

import os
import sys
import time

import numpy as np

KERNEL = os.environ["BENCH_KERNEL"]
N = int(os.environ["BENCH_N"])

rng = np.random.default_rng(42)


def kernel_matmul():
    A = rng.random((N, N))
    B = rng.random((N, N))
    return lambda: A @ B


def kernel_solve():
    A = rng.random((N, N)) + N * np.eye(N)
    b = rng.random(N)
    return lambda: np.linalg.solve(A, b)


def kernel_lu():
    # NumPy doesn't expose a standalone LU; SciPy's `linalg.lu` is the
    # NumPy-stack analogue but has hard NumPy ABI compatibility issues
    # with newer NumPy. Solve uses LU internally and is on the bench
    # already as `solve` — skip the standalone LU comparison.
    import sys
    print("lu N=%d best=null" % N, file=sys.stderr)
    sys.exit(0)


def kernel_qr():
    A = rng.random((N, N))
    return lambda: np.linalg.qr(A)


def kernel_chol():
    A = rng.random((N, N))
    S = A.T @ A + np.eye(N)
    return lambda: np.linalg.cholesky(S)


def kernel_inv():
    A = rng.random((N, N)) + N * np.eye(N)
    return lambda: np.linalg.inv(A)


def kernel_eig():
    A = rng.random((N, N))
    S = (A + A.T) / 2
    return lambda: np.linalg.eigvalsh(S)


def kernel_svd():
    A = rng.random((N, N))
    return lambda: np.linalg.svd(A, compute_uv=False)


KERNELS = {
    "matmul": kernel_matmul,
    "solve": kernel_solve,
    "lu": kernel_lu,
    "qr": kernel_qr,
    "chol": kernel_chol,
    "inv": kernel_inv,
    "eig": kernel_eig,
    "svd": kernel_svd,
}

setup = KERNELS[KERNEL]
fn = setup()

best = float("inf")
for _ in range(3):
    t0 = time.perf_counter_ns()
    fn()
    t1 = time.perf_counter_ns()
    best = min(best, (t1 - t0) / 1e9)

print(f"{KERNEL} N={N} best={best:.9f} s")
