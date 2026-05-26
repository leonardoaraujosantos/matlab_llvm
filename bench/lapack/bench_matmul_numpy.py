#!/usr/bin/env python3
"""NumPy matmul wall-clock — companion to bench_matmul.m."""

import os
import sys
import time

import numpy as np

N = int(os.environ["BENCH_N"])

rng = np.random.default_rng(42)
A = rng.random((N, N))
B = rng.random((N, N))

best = float("inf")
for _ in range(3):
    t0 = time.perf_counter_ns()
    C = A @ B
    t1 = time.perf_counter_ns()
    best = min(best, (t1 - t0) / 1e9)

print(f"matmul N={N} best={best:.9f} s", file=sys.stderr)
print(f"matmul N={N} best={best:.9f} s")
