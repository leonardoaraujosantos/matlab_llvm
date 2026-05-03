% pdepe with Neumann (zero-flux) boundary conditions:
%   u_t = u_xx,  u_x(0,t) = u_x(1,t) = 0,  u(x, 0) = cos(pi*x).
% Analytic: u(x, t) = exp(-pi^2 * t) * cos(pi*x).
%
% This exercises the Neumann/Robin path: ql = qr = 1, so the solver
% can't eliminate boundary values from the state vector. Boundary
% nodes evolve under the half-cell discretization with f = -pl/ql at
% the boundary face.

m = 0;
pdefun = @(x,t,u,dudx) [1; dudx; 0];
icfun  = @(x) cos(3.141592653589793 * x);
% Neumann zero on both ends: pl = pr = 0, ql = qr = 1 → f = 0 at faces.
bcfun  = @(xl,ul,xr,ur,t) [0; 1; 0; 1];

xmesh = linspace(0, 1, 21);
tspan = [0 0.1];

sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan);
disp(size(sol, 2));    % must equal numel(xmesh) = 21

% Boundary values track exp(-pi^2 * 0.1) ≈ 0.37257.
ya = exp(0 - 9.869604401089358 * 0.1);

% u(0, 0.1) ≈ 0.37257
err_l = abs(sol(end, 1) - ya);
if err_l < 0.01; disp(1); else; disp(0); end

% u(1, 0.1) ≈ -0.37257
err_r = abs(sol(end, end) - (0 - ya));
if err_r < 0.01; disp(1); else; disp(0); end

% Symmetry: u(x, t) + u(1-x, t) = 0 since cos(pi*x) + cos(pi*(1-x)) = 0.
err_sym = abs(sol(end, 6) + sol(end, 16));
if err_sym < 0.01; disp(1); else; disp(0); end

% Mid-point u(0.5, t) is identically 0 (cos(pi/2) = 0). Numerical
% drift sits at FP rounding (~1e-15).
if abs(sol(end, 11)) < 1e-10; disp(1); else; disp(0); end
