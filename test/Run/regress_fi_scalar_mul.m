% regress_fi_scalar_mul.m — #294: multiplying a fi (fixed-point) value by a
% plain numeric scalar. The scalar operand is encoded as a wide fixed-point
% constant, so the integer product overflowed the work word (it was sized to
% the OUTPUT word length, not the lhs.WL + rhs.WL the product needs) and the
% real-world result scaled back to 0. The fi multiply now sizes the work
% integer to hold the full product. Also exercises fi propagation across a
% user-function boundary (the issue's repro: stage(x) = x * 2).

a = fi(0.5, 1, 16, 14);

disp(double(a * 2));        % 1
disp(double(2 * a));        % 1
disp(double(a * 2.0));      % 1
disp(double(a * 3));        % 1.5
disp(double(a * a));        % 0.25  (fi * fi, unchanged)

% fi propagates across a user function; the result stays fi-typed, so a
% following fi + fi keeps fixed-point semantics.
b = stage(a);               % 0.5 * 2 = 1.0, fi-typed
disp(double(b));            % 1
disp(double(b + a));        % 1.5

function y = stage(x)
  y = x * 2;
end
