% bench_qr.m — [Q,R] = qr(A) wall-clock. Phase 2 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N);
best = Inf;
for trial = 1:3
    tic; [Q, R] = qr(A); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('qr N=%d best=%.9f s\n', N, best);
