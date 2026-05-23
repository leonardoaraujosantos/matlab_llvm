% Reductions over a matrix-valued struct field (field read must fetch a matrix).
s.v = [1 2 3 4 5];
s.w = [10 20 30];
fprintf('sum_v=%d\n', sum(s.v));
fprintf('numel_v=%d\n', numel(s.v));
fprintf('mean_v=%.4f\n', mean(s.v));
fprintf('sum_w=%d\n', sum(s.w));
