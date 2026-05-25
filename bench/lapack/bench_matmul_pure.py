#!/usr/bin/env python3
"""Pure-Python triple-loop matmul — the "what-if-you-wrote-it-yourself" baseline."""

import os
import random
import sys
import time

N = int(os.environ["BENCH_N"])

random.seed(42)
A = [[random.random() for _ in range(N)] for _ in range(N)]
B = [[random.random() for _ in range(N)] for _ in range(N)]


def matmul(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C


best = float("inf")
for _ in range(3):
    t0 = time.perf_counter_ns()
    C = matmul(A, B, N)
    t1 = time.perf_counter_ns()
    best = min(best, (t1 - t0) / 1e9)

print(f"matmul N={N} best={best:.9f} s")
