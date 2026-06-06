% Regression for #235: numeric factor(n) — prime factorisation as an
% ascending row vector with multiplicity (prod(factor(n)) == n). This now
% works alongside the Symbolic Math factor(expr, var); the names collide and
% the frontend splits them by argument type (numeric -> matlab_factor, sym ->
% matlab_sym_factor). Printed via fprintf %.0f so the output is byte-identical
% across all four execute backends.
f = factor(60);
fprintf('%.0f %.0f %.0f %.0f\n', numel(f), f(1), f(3), f(4));
fprintf('%.0f %.0f\n', sum(factor(360)), prod(factor(360)));
fprintf('%.0f\n', factor(17));
g = factor(1);
fprintf('%.0f %.0f\n', numel(g), g(1));
