% Basic descriptive statistics on a column of measurements. We use
% disp() rather than fprintf() for the computed scalars because format
% specifiers like %.3f only kick in for values whose type Sema has
% already inferred — aggregate calls like mean/min/max return `any`,
% so disp() is the safer path for now.
x = [12.1 11.8 13.0 12.7 12.5 11.9 12.3 12.8 13.2 12.0];

disp('x ='); disp(x);

disp('n      ='); disp(numel(x));
disp('sum    ='); disp(sum(x));
disp('mean   ='); disp(mean(x));
disp('min    ='); disp(min(x));
disp('max    ='); disp(max(x));
disp('range  ='); disp(max(x) - min(x));

% 2-norm via sqrt(sum of squares).
disp('norm   ='); disp(sqrt(sum(x .* x)));
