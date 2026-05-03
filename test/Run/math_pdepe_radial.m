% pdepe with cylindrical (m=1) and spherical (m=2) symmetry. The PDE
% form picks up x^m factors:
%   c · ∂u/∂t = (1/x^m) · ∂/∂x [x^m · f(x,t,u,∂u/∂x)] + s
%
% Cylindrical Laplacian on annulus r ∈ [1, 2] with u(1) = 0, u(2) = 1
% has steady-state u(r) = log(r)/log(2). The spherical analog gives
% u(r) = 2 - 2/r.

pdefun = @(x,t,u,dudx) [1; dudx; 0];
icfun  = @(r) r - 1;
bcfun  = @(xl,ul,xr,ur,t) [ul; 0; ur - 1; 0];

xmesh = linspace(1, 2, 21);
tspan = [0 0.5 5];

% --- Cylindrical (m = 1) ---
sol_c = pdepe(1, pdefun, icfun, bcfun, xmesh, tspan);
% At r = 1.5: analytic log(1.5)/log(2) ≈ 0.5850.
err_c = abs(sol_c(end, 11) - log(1.5) / log(2));
if err_c < 0.001; disp(1); else; disp(0); end

% --- Spherical (m = 2) ---
sol_s = pdepe(2, pdefun, icfun, bcfun, xmesh, tspan);
% At r = 1.5: analytic 2 - 2/1.5 = 2/3.
err_s = abs(sol_s(end, 11) - (2 - 2 / 1.5));
if err_s < 0.001; disp(1); else; disp(0); end

% Boundaries hit Dirichlet exactly.
disp(sol_c(end, 1));
disp(sol_c(end, end));
disp(sol_s(end, 1));
disp(sol_s(end, end));

% m outside {0, 1, 2}, or m > 0 with xmesh(1) = 0, returns empty.
sol_bad = pdepe(3, pdefun, icfun, bcfun, xmesh, tspan);
disp(numel(sol_bad));    % 0

xmesh0 = linspace(0, 1, 21);
sol_bad2 = pdepe(2, pdefun, icfun, bcfun, xmesh0, tspan);
disp(numel(sol_bad2));   % 0
