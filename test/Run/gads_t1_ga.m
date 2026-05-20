% Global Optimization Tier-1 — genetic algorithm on Rastrigin.
% Rastrigin (n=2) has ~30 local minima on [-5.12,5.12]^2 and a single
% global min at the origin (f=0).  ga finds the global basin; the
% fmincon hybrid polish drives it to f=0.  (x(1)*x(1) avoids the
% scalar-matpow-in-anon compiler gap.)
rng(42);
ras = @(x) 20 + (x(1)*x(1) - 10*cos(2*pi*x(1))) + (x(2)*x(2) - 10*cos(2*pi*x(2)));
lb = [-5.12; -5.12];  ub = [5.12; 5.12];
x = ga(ras, 2, [], [], [], [], lb, ub);
fprintf('ga f = %.4f\n', ras(x));        % 0.0000 (global min)
fprintf('nvars = %.0f\n', size(x, 1));   % 2
