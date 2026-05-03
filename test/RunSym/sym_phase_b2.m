% Phase 6.2 — sym('pi') singleton, simplify+refine, matrix literal
% sym detection, variadic sym_solve_sys, multi-condition IVP.

% sym('pi') resolves to the SymPP Pi singleton (not Symbol("pi")).
disp(vpa(sym('pi'), 32))
syms x
disp(sin(sym('pi')))

% simplify auto-honours assumptions via refine().
syms y
assume(y, 'positive')
disp(simplify(sqrt(y*y)))

% Standard matrix literal `[a 1; 2 b]` routes through symmat when any
% entry is sym (no longer needs the explicit sym_matrix(R, C, ...) form).
syms a b
M = [a, sym(1); sym(2), b];
disp(M)
disp(sym_det(M))

% Variadic sym_solve_sys for systems of any size.
syms u v w
sols = sym_solve_sys([u^2 + v^2 - w, v - u, w - sym(2)], [u, v, w]);
disp(sols)

% Multi-condition IVP — pass parallel arrays of x and y values.
syms t yt yp
sol = dsolve(yp + yt, yt, yp, t);
disp(apply_ivp(sol, t, [sym(0)], [sym(1)]))
