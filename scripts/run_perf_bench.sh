#!/usr/bin/env bash
# scripts/run_perf_bench.sh — Tier 7 (acceleration_roadmap §8) perf gate.
#
# Wraps bench/lapack/driver.sh with the CI-friendly slice (small N, no
# pure-Python, no gpu_gemm) and produces a results JSON whose path is
# echoed on stdout.
#
# Used by the perf-bench CI lane to bench HEAD and the merge-base on
# the same runner, then diff via bench/lapack/check_regression.py.
# The driver builds the runtime objects fresh each invocation so the
# bench reflects the current working-tree state — no stale caches.
#
# Usage:
#   bash scripts/run_perf_bench.sh <tag> [<matlabc>]
#
# `<tag>` is the result filename suffix (e.g. "head" or "base").
# `<matlabc>` defaults to `build/matlabc`.

set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "usage: $0 <tag> [<matlabc>]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATLABC="${2:-$ROOT/build/matlabc}"
if [[ ! -x "$MATLABC" ]]; then
  echo "error: matlabc not found at $MATLABC" >&2
  exit 2
fi

# CI slice — fast subset of the full local sweep.  Same kernels as the
# regression-relevant LAPACK + scalar-loop set; one size (N=300) which
# is past the BLAS threshold for matmul/solve/etc.; matlab_llvm and
# numpy only (skipping pure-Python's multi-second N=1000 runs).
# gpu_gemm omitted: Metal isn't available on Linux CI runners and the
# CPU fallback is identical to matmul.
export BENCH_KERNELS="${BENCH_KERNELS:-matmul solve lu qr chol inv eig svd mandelbrot}"
export BENCH_SIZES="${BENCH_SIZES:-300}"
export BENCH_IMPLS="${BENCH_IMPLS:-matlab_llvm numpy}"

export MATLABC

bash "$ROOT/bench/lapack/driver.sh" "perf_$TAG" >&2
echo "$ROOT/bench/lapack/results/perf_$TAG.json"
