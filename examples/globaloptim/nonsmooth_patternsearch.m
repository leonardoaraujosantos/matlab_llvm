% nonsmooth_patternsearch.m — Global Optimization Toolbox Tier-3 headline.
%
% Direct search (User's Guide, Using Direct Search chapter).
% patternsearch is derivative-free: it polls a positive-spanning basis
% {±e_i} on a mesh, moving to any better point and refining the mesh,
% never touching a gradient.  That makes it robust on objectives where
% gradients are undefined, discontinuous, or noisy — exactly where a
% gradient-based solver fails.
%
%   patternsearch(fun, x0, A, b, Aeq, beq, lb, ub)
%
% This demo pits patternsearch against the gradient-based fminunc on a
% *discontinuous* staircase bowl: the objective is a paraboloid quantized
% into flat steps, so its finite-difference gradient is ~0 almost
% everywhere.  The gradient solver stalls at its start point; the direct
% search steps down the staircase to the global minimum.

% ----- Discontinuous staircase bowl -----------------------------------
% f = round( 2*((x1-2)^2 + (x2+3)^2) ) / 2   — global min f=0 at (2,-3).
% (x*x products sidestep the scalar-matpow-in-anonymous-function gap.)
staircase = @(x) round( ((x(1)-2)*(x(1)-2) + (x(2)+3)*(x(2)+3)) * 2 ) / 2;

lb = [-10; -10];
ub = [ 10;  10];
x0 = [7; 7];

% ----- Gradient-based solver stalls -----------------------------------
xu = fminunc(staircase, x0);
fprintf('Gradient solver (fminunc) from (7, 7):\n');
fprintf('  f = %.4f   (stalled — FD gradient is 0 on every flat step)\n\n', ...
        staircase(xu));

% ----- Direct search steps to the global minimum ----------------------
xp = patternsearch(staircase, x0, [], [], [], [], lb, ub);
fprintf('Direct search (patternsearch):\n');
fprintf('  f = %.4f   at (%.2f, %.2f)   (global)\n\n', ...
        staircase(xp), xp(1), xp(2));

% ----- Bonus: a nonsmooth V-valley with the min at a kink -------------
% f = 3|x1-1| + 3|x2+2| + 0.5  — nondifferentiable at the minimum (1,-2).
vvalley = @(x) 3*abs(x(1) - 1) + 3*abs(x(2) + 2) + 0.5;
xv = patternsearch(vvalley, [5; 5], [-10; -10], [10; 10]);
fprintf('Nonsmooth V-valley (kinked minimum):\n');
fprintf('  f = %.4f   at (%.2f, %.2f)\n', vvalley(xv), xv(1), xv(2));
