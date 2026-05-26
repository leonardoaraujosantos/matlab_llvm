% Phase 4 of lapack_roadmap §4 — GPU library replacement for GEMM.
% Same gemm shape as bench_matmul, but routes through gpucoder.gemm
% which dispatches to MPSMatrixMultiplication when
% MATLAB_GPU_TARGET=metal is set in the environment (falls back to
% the host BLAS path otherwise).
%
% The fp64 → fp32 → MPS → fp32 → fp64 round-trip is intentionally
% hidden inside matlab_gpu_metal_gemm_double; the user-level surface
% is the same fp64 matlab_mat * matlab_mat call shape.
N = __BENCH_N__;
A = rand(N, N);
B = rand(N, N);
best = Inf;
for trial = 1:3
    tic;
    C = gpucoder.gemm(A, B);
    elapsed = toc;
    if elapsed < best
        best = elapsed;
    end
end
fprintf('gpu_gemm N=%d best=%.9f s\n', N, best);
