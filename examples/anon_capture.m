% Anonymous functions that close over values from the enclosing scope.
% Captured values are snapshotted at the moment the `@(x) ...` is
% created, matching MATLAB's by-value capture semantics — reassigning
% the captured variable afterwards does not alter the handle's behavior.

% Add-a-constant closure.
k = 5;
f = @(x) x + k;
disp('f = @(x) x + 5');
disp('f(3)  ='); disp(f(3));
disp('f(10) ='); disp(f(10));

% Affine closure: two captures in the same expression.
a = 2;
b = 3;
g = @(x) a * x + b;
disp('g = @(x) 2*x + 3');
disp('g(5)  ='); disp(g(5));
disp('g(10) ='); disp(g(10));

% Capture-by-value proof: reassign k, the old handle still uses 5.
k = 100;
disp('after k = 100, f(0) still yields:');
disp(f(0));

% Matrix capture: A is captured by pointer at @-time.
A = [1 2 3; 4 5 6; 7 8 9];
diagi = @(i) A(i, i);
disp('diagonal of A via matrix-capturing handle:');
disp(diagi(1));
disp(diagi(2));
disp(diagi(3));

% Matrix capture driving a matrix result.
M = [1 2; 3 4];
scaled = @(s) M * s;
disp('M scaled by 3:');
disp(scaled(3));
