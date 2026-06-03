% regress_logical_in_arithmetic.m — regression test for using a scalar
% logical / comparison result (an i1) in arithmetic (#161). Before the fix,
% an i1 operand feeding matlab.add/sub/mul(matmul)/div mismatched the f64
% other operand and the op was left unconverted ("1× matlab.add left in
% IR"). MATLAB promotes a logical to double in arithmetic, so an i1 operand
% now widens to f64 (UIToFP — 0/1) in the scalar arith lowering. (#152 was
% the disp analog.)

% --- logical/comparison result in +, -, *, / ----------------------
x = 1 | 0;
disp(x + 10);        % 11
disp((5 > 0) + 10);  % 11
disp((3 > 5) + 10);  % 10  (false -> 0)
disp((5 > 0) - 1);   % 0
disp((5 > 0) * 3);   % 3
disp(3 * (2 > 1));   % 3   (other operand order)
disp((5 > 0) / 2);   % 0.5
disp(~0 + 5);        % 6

% --- logical + logical (both i1) -> double -------------------------
disp((1 | 0) + (1 & 1));   % 2

% --- plain numeric arithmetic is unaffected ------------------------
disp(2 + 3);         % 5
disp(2 * 3);         % 6
disp(6 / 2);         % 3
