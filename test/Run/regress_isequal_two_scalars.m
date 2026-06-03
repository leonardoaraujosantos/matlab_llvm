% regress_isequal_two_scalars.m — regression test for isequal with two
% scalar arguments (#155). Before the fix the LowerTensorOps pde_table had
% isequal only as `pp` (two matrix ptrs → matlab_isequal); two scalar f64
% args gave an `ff` shape that matched nothing → "unsupported call shape".
% Added an `ff` entry → matlab_isequal_2s (returns a bare f64 0/1, since
% isequal of scalars is genuinely scalar). Same class as the max/min
% two-scalar gap (#153). Matrix isequal is unaffected.

% --- two scalars ---------------------------------------------------
disp(isequal(5, 5));        % 1
disp(isequal(3, 4));        % 0
disp(isequal(-2, -2));      % 1
disp(isequal(0, 0));        % 1

% --- scalar variables ----------------------------------------------
a = 7; b = 7; c = 9;
disp(isequal(a, b));        % 1
disp(isequal(a, c));        % 0

% --- result usable in a guarded condition --------------------------
if isequal(2, 2) == 1
  disp(42);                 % 42
end

% --- matrix isequal is unaffected ----------------------------------
disp(isequal([1 2 3], [1 2 3]));  % 1
disp(isequal([1 2], [1 3]));      % 0
