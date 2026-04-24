% linspace(a, b, n) — evenly spaced points, endpoints inclusive.
% `linspace(a, b)` defaults n=100, which the single-argument form
% doesn't reach yet (needs a Lowerer default); only the 3-arg
% shape is shipped.
disp(linspace(0, 1, 5));
disp(linspace(-2, 2, 5));
disp(linspace(0, 10, 11));
