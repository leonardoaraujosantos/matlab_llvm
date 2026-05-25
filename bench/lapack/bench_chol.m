% bench_chol.m — R = chol(A'*A + I) wall-clock (SPD input). Phase 2 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N);
S = A' * A + eye(N);    % SPD
best = Inf;
for trial = 1:3
    tic; R = chol(S); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('chol N=%d best=%.9f s\n', N, best);
