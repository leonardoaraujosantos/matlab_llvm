% Symbolic Math Toolbox (SymPP) smoke test — guards the symbolic builtins
% and the AOT sym link recipe. Linked via the `.requires-sym` marker; the
% harness skips it when SymPP / the WITH_SYM runtime object is unavailable.
syms x a b c
f = a*x^2 + b*x + c;
disp(diff(f, x));         % first derivative wrt x
disp(diff(f, x, 2));      % second derivative
disp(int(x^2, x));        % indefinite integral
disp(subs(f, x, 2));      % substitution x -> 2
