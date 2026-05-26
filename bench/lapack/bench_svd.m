% bench_svd.m — svd(A) wall-clock. Phase 3 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N);
best = Inf;
for trial = 1:3
    tic; sv = svd(A); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('svd N=%d best=%.9f s\n', N, best);
