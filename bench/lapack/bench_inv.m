% bench_inv.m — inv(A) wall-clock. Phase 2 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N) + N * eye(N);
best = Inf;
for trial = 1:3
    tic; X = inv(A); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('inv N=%d best=%.9f s\n', N, best);
