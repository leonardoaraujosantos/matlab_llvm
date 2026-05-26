% bench_solve.m — A \ b wall-clock. Phase 2 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N) + N * eye(N);   % well-conditioned
b = rand(N, 1);
best = Inf;
for trial = 1:3
    tic; x = A \ b; elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('solve N=%d best=%.9f s\n', N, best);
