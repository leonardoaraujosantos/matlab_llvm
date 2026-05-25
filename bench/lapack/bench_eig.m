% bench_eig.m — eig of a symmetric matrix. Phase 3 kernel.
N = __BENCH_N__;
rng(42);
A = rand(N, N);
S = (A + A') / 2;   % symmetric — symmetric path is what we accelerate
best = Inf;
for trial = 1:3
    tic; e = eig(S); elapsed = toc;
    if elapsed < best; best = elapsed; end
end
fprintf('eig N=%d best=%.9f s\n', N, best);
