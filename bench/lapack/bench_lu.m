% bench_lu.m — [L,U] = lu(A) wall-clock. Phase 2 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N) + N * eye(N);
best = Inf;
for trial = 1:3
    tic; [L, U] = lu(A); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('lu N=%d best=%.9f s\n', N, best);
