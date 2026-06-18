% Regression (examples/gpu/benchmark_gpu_backend.m): a trailing dtype
% class-name string on a shape builtin — rand/randn/zeros/ones/eye(dims...,
% 'single'|'double') — must be accepted. Before the fix this raised
% "unsupported call shape for built-in function 'rand': 3 arguments, 1
% return value". LowerTensorOps now strips the dtype const_char (the CPU
% lane is double-only, so single/double share storage), gated on the arg
% being a const_char so the 3-D zeros(n, m, d) size form (numeric third
% arg) keeps its own path.

% Deterministic checks (zeros / ones / eye).
Z = zeros(2, 3, 'single');
fprintf('z = %d %d\n', size(Z));
O = ones(2, 2, 'double');
fprintf('osum = %g\n', sum(O(:)));
I = eye(3, 'single');
fprintf('itrace = %g\n', sum(diag(I)));

% rand / randn are non-deterministic; check shape + range only.
R = rand(4, 4, 'single');
fprintf('rok = %d rn = %d\n', all(R(:) >= 0 & R(:) <= 1), numel(R));

% 1-arg + dtype form folds through the eye(n) -> eye(n, n) normalizer.
R2 = rand(5, 'single');
fprintf('r2n = %d\n', numel(R2));

% 3-D numeric zeros must still work (third arg is a size, not a dtype tag).
Z3 = zeros(2, 3, 4);
fprintf('z3 = %d\n', numel(Z3));
