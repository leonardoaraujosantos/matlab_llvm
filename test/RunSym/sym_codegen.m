% simplify / expand / factor / vpa / double — symbolic transforms +
% numeric evaluation (deterministic, no pointer-printing builtins).
syms x
disp(simplify((x^2 - 1)/(x - 1)))    % SymPP rational form
disp(expand((x + 2)^2))              % x^2 + 4*x + 4
disp(factor(x^2 - 4, x))             % (x-2)(x+2)
disp(vpa(sym('pi'), 16))             % 3.141592653589793
disp(double(sym(3)/sym(4)))          % 0.75
