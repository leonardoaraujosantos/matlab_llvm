% coneprog — second-order cone programming (Optimization Toolbox
% Tier-3).
%
% coneprog minimises f'*x subject to a second-order cone constraint
% ||Asc*x + bsc|| <= dsc'*x + gamma plus linear constraints.
% matlab_llvm reformulates the cone as a single nonlinear inequality
% and routes it through the augmented-Lagrangian core.
%
%   x = coneprog(f, Asc, bsc, dsc, gamma, A, b, Aeq, beq, lb, ub)

% --- 1. maximise x1 + x2 on the unit disk ------------------------
%   minimise -x1 - x2  s.t.  ||x|| <= 1   (Asc = I, bsc = 0, dsc = 0,
%   gamma = 1).  The optimum is x = [1; 1] / sqrt(2) ~ [0.7071;0.7071].
f   = [-1; -1];
Asc = [1, 0; 0, 1];
bsc = [0; 0];
dsc = [0; 0];
x = coneprog(f, Asc, bsc, dsc, 1);
fprintf('max x1+x2 on unit disk: x = [%.4f, %.4f]\n', x(1), x(2));
fprintf('  ||x|| = %.4f  (cone active at 1)\n', sqrt(x(1)*x(1) + x(2)*x(2)));

% --- 2. minimise -x1 alone on the unit disk ----------------------
%   The optimum slides to x = [1; 0].
y = coneprog([-1; 0], Asc, bsc, dsc, 1);
fprintf('max x1 on unit disk:    x = [%.4f, %.4f]\n', y(1), y(2));

% --- 3. a smaller cone, gamma = 0.5 ------------------------------
%   ||x|| <= 0.5  →  x = [0.5; 0.5] / sqrt(2) ~ [0.3536; 0.3536].
z = coneprog(f, Asc, bsc, dsc, 0.5);
fprintf('max x1+x2, radius 0.5:  x = [%.4f, %.4f]\n', z(1), z(2));
