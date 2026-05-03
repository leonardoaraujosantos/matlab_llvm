% pdepe — 1-D heat equation u_t = u_xx on [0, 1], with
% u(0, t) = u(1, t) = 0 (Dirichlet zero) and u(x, 0) = sin(π·x).
%
% Analytic solution: u(x, t) = exp(-π²·t) · sin(π·x). The peak at
% x = 0.5 decays from 1 (at t = 0) to exp(-0.0987) ≈ 0.906 at
% t = 0.01 and to exp(-0.987) ≈ 0.373 at t = 0.1.
%
% Internally `pdepe` discretises space via finite differences on the
% supplied `xmesh` and hands the resulting interior ODE system to
% `ode23s` for stiff time integration (the spatial discretisation makes
% even simple parabolic problems stiff at moderate mesh sizes).

m       = 0;
pdefun  = @(x, t, u, dudx) [1; dudx; 0];          % c=1, f=du/dx, s=0
icfun   = @(x) sin(3.141592653589793 * x);
bcfun   = @(xl, ul, xr, ur, t) [ul; 0; ur; 0];    % Dirichlet zero ends

xmesh = linspace(0, 1, 21);
tspan = [0 0.01 0.05 0.1];                       % user-grid output times

sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan);

disp('Heat equation u_t = u_xx, u(x,0) = sin(pi*x), u(0,t) = u(1,t) = 0');
disp('  number of output time points:');
disp(size(sol, 1));
disp('  mesh points:');
disp(size(sol, 2));
disp('  peak u(0.5, t) at the four output times:');
disp(sol(1, 11));    % t=0     → analytic 1
disp(sol(2, 11));    % t=0.01  → analytic 0.9061
disp(sol(3, 11));    % t=0.05  → analytic 0.6105
disp(sol(4, 11));    % t=0.1   → analytic 0.3727
