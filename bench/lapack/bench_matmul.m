% bench_matmul.m — matmul wall-clock for the LAPACK acceleration epic.
%
% Reads N from the BENCH_N env var (set by driver.sh). Builds deterministic
% inputs from a fixed seed, runs `A * B` three times, prints the minimum.

N = __BENCH_N__;

rng(42);
A = rand(N, N);
B = rand(N, N);

best = Inf;
for trial = 1:3
    tic;
    C = A * B;
    elapsed = toc;
    if elapsed < best
        best = elapsed;
    end
end

% Print only the minimum-trial time in seconds, full precision.
% Driver parses the last numeric token on the last line.
fprintf('matmul N=%d best=%.9f s\n', N, best);
