#!/usr/bin/env python3
"""Pure-Python Mandelbrot — scalar inner loop, identical algorithm to bench_mandelbrot.m."""

import os
import sys
import time

N = int(os.environ["BENCH_N"])
max_iter = 100
re_min, re_max = -2.0, 1.0
im_min, im_max = -1.5, 1.5


def mandelbrot(N: int, max_iter: int) -> None:
    counts = [[0] * N for _ in range(N)]
    for py in range(N):
        cim = im_min + (im_max - im_min) * py / (N - 1)
        for px in range(N):
            cre = re_min + (re_max - re_min) * px / (N - 1)
            zre, zim = 0.0, 0.0
            count = 0
            for k in range(1, max_iter + 1):
                zre2 = zre * zre - zim * zim + cre
                zim2 = 2.0 * zre * zim + cim
                zre, zim = zre2, zim2
                if zre * zre + zim * zim > 4.0:
                    break
                count = k
            counts[py][px] = count


best = float("inf")
for _ in range(3):
    t0 = time.perf_counter_ns()
    mandelbrot(N, max_iter)
    t1 = time.perf_counter_ns()
    best = min(best, (t1 - t0) / 1e9)

print(f"mandelbrot N={N} best={best:.9f} s")
