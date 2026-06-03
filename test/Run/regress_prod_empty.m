% Regression: prod of an empty matrix is the multiplicative identity 1
% (MATLAB), not 0. The empty reduction previously hard-coded 0.0 for all
% reductions, which is correct for sum but wrong for prod.
fprintf('empty=%.0f\n', prod([]));
fprintf('normal=%.0f\n', prod([2 3 4]));
fprintf('col=%.0f\n', prod([2;3;4]));
m = prod([1 2; 3 4]);          % column-wise -> [3 8]
fprintf('mat=%.0f %.0f\n', m(1), m(2));
