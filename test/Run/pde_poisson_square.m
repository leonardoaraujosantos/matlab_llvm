% pde_poisson_square.m — 2-D Poisson equation FEM smoke test.
%
% Solves  -Lap(u) = 1   on the unit square [0,1]x[0,1]
%             u = 0   on the boundary.
%
% Validates the Tier-1 pde_mesh_rect_tri -> pde_assemble_poisson_2d ->
% pde_apply_dirichlet -> mldivide pipeline against the known analytic
% peak value u(0.5, 0.5) ~= 0.0736713.  See
% docs/pde_toolbox_roadmap.md section 2.

mesh = pde_mesh_rect_tri(0.0, 1.0, 0.0, 1.0, 21, 21);
bnd  = pde_boundary_nodes_rect(mesh);

sys  = pde_assemble_poisson_2d(mesh, 1.0, 0.0, 1.0);   % c=1, a=0, f=1
sys2 = pde_apply_dirichlet(sys, bnd, 0.0);

K = pde_sys_K(sys2);
F = pde_sys_F(sys2);
u = K \ F;

% Centre node is at (0.5, 0.5) — for a 21x21 grid that's node id
% 10 + 10*21 + 1 = 221.  The peak of the FEM solution is at this same
% node so u(221) doubles as u_peak.
u_center = u(221);

% Format to fewer decimals so floating-point jitter doesn't fail the
% gold comparison.
fprintf('PDE Tier-1: -Lap(u) = 1 on [0,1]x[0,1], u=0 on bnd\n');
fprintf('  grid:         21 x 21 (Nn=441, Nt=800)\n');
fprintf('  u(0.5, 0.5) = %.4f\n', u_center);
fprintf('  analytic    = 0.0737\n');
