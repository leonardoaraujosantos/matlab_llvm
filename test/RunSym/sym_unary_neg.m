% Regression for #241: unary minus on a sym function-call or sym sub-expression
% must route to matlab_sym_neg (it used to fall through to the numeric
% matlab_neg_m and segfault on a sym pointer).
syms theta x
disp(-sin(theta))           % function-call operand (was a segfault)
disp(-cos(theta))           % function-call operand
disp(simplify(-sin(theta))) % inside simplify
disp(-exp(2*x))             % nested non-trivial argument
disp(-theta)                % bare symbol (already worked — guard)
disp(-(theta + sym(1)))     % sym sub-expression (already worked — guard)
