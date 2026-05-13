% pde_nonlinear.m — Tier-4 nonlinear FEM smoke test.
%
% Solves the nonlinear Poisson equation:
%   -div(c(u) * grad(u)) = f    on the unit square
%      u = 0 on the boundary
% where c(u) = c0 * (1 + alpha * u^2) is a u-dependent diffusivity.
%
% Validates the Picard iteration outer loop in
% pde_solve_nonlinear_2d.  See docs/pde_toolbox_roadmap.md section 5.

mesh = pde_mesh_rect_tri(0.0, 1.0, 0.0, 1.0, 21, 21);

% c0 = 1, alpha = 5, f = 1, c_func = 1 (quadratic).
result = pde_solve_nonlinear_2d(mesh, 1.0, 5.0, 1.0, 1.0);
u    = pde_result_solution(result);
iter = pde_result_num_iters(result);

% Centre node (i=10, j=10) → 10*21 + 10 + 1 = 221.
u_center = u(221);

fprintf('PDE Tier-4: -div(c(u)*grad u) = 1, c(u) = 1 + 5*u^2\n');
fprintf('  mesh:               21 x 21 (441 nodes, 800 tris)\n');
fprintf('  Picard iters:       %.0f\n', iter);
fprintf('  u(0.5, 0.5):        %.4f\n', u_center);
fprintf('  linear u(0.5,0.5):  0.0737  (alpha = 0 baseline)\n');
