% pdepe — 1-D heat equation u_t = u_xx on [0, 1] with Dirichlet zero
% boundary conditions and u(x, 0) = sin(pi*x). Analytic solution
%   u(x, t) = exp(-pi^2 * t) * sin(pi*x).
%
% This is the textbook test for any 1-D parabolic PDE solver. It also
% exercises the full pdepe stack:
%   - Sema registration of `pdepe`
%   - LowerTensorOps single-result 6-arg dispatch
%   - LowerAnonCalls outlining of three anon-fn handles with vector
%     return shapes (pdefun, bcfun)
%   - The runtime method-of-lines wrapper around ode23s_v.

m = 0;
pdefun = @(x,t,u,dudx) [1; dudx; 0];          % c=1, f=du/dx, s=0
icfun  = @(x) sin(3.141592653589793 * x);
bcfun  = @(xl,ul,xr,ur,t) [ul; 0; ur; 0];      % Dirichlet zero both ends

xmesh = linspace(0, 1, 21);
tspan = [0 0.1];

sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan);

% Output shape: Nt × Nx, with sol(end, :) the field at t = 0.1.
disp(size(sol, 2));    % must equal numel(xmesh) = 21

% Numerical vs analytic at x = 0.5 (mesh index 11):
%   analytic = exp(-pi^2 * 0.1) ≈ 0.37257
ya = exp(0 - 9.869604401089358 * 0.1);

% Error should be well under 1e-2 on a 21-point mesh.
err = abs(sol(end, 11) - ya);
if err < 0.01; disp(1); else; disp(0); end

% Symmetry: the solution preserves sin-shape symmetry around x = 0.5.
err_sym = abs(sol(end, 6) - sol(end, 16));
if err_sym < 0.01; disp(1); else; disp(0); end

% Energy decay: ||u(t)||_inf at t=0.1 < ||u(0)||_inf = 1.
peak_t = abs(sol(end, 11));
if peak_t < 0.5; disp(1); else; disp(0); end
