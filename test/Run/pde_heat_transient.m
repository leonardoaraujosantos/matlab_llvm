% pde_heat_transient.m — Tier-3 transient parabolic FEM smoke test.
%
% Solves the heat equation on the unit square:
%   d_u/d_t = Lap(u)    on [0,1] x [0,1]
%   u = 0   on the boundary
%   u(x, y, 0) = 1 inside, 0 on boundary
%
% Forward-Euler time stepping on the FEM-assembled (M, K, F) system.
% u should decay toward zero as the source is absorbed at the Dirichlet
% edges.  See docs/pde_toolbox_roadmap.md section 4.

mesh = pde_mesh_rect_tri(0.0, 1.0, 0.0, 1.0, 11, 11);
bnd  = pde_boundary_nodes_rect(mesh);

% Assemble M, K, F for d=1, c=1, a=0, f=0.
sys  = pde_assemble_transient_2d(mesh, 1.0, 0.0, 0.0);
M = pde_sys_M(sys);
K = pde_sys_K(sys);
F = pde_sys_F(sys);

% Initial condition: u = 1 interior, 0 on boundary.
u = pde_init_uniform_2d(mesh, 1.0, bnd);

% Step forward.  Forward Euler stability requires dt < 2/lambda_max
% of M^-1 K; for an 11x11 grid that's ~1e-3.  Use 5e-4 with 400 steps
% to reach t = 0.2.
dt    = 5e-4;
nstep = 400;
for i = 1:nstep
    u = pde_step_forward_euler_2d(M, K, F, u, bnd, dt);
end

% Centre node (i=5, j=5): idx = 5 + 5*11 = 60, 1-based = 61.
u_center = u(61);
u_init   = 1.0;

fprintf('PDE Tier-3: heat eq u_t = Lap(u), u=0 on bnd, u(x,y,0)=1\n');
fprintf('  mesh:                 11 x 11 (121 nodes, 200 tris)\n');
fprintf('  dt = 5e-4, steps = 400, total t = 0.2\n');
fprintf('  u_center(t=0.0) = %.4f\n', u_init);
fprintf('  u_center(t=0.2) = %.4f\n', u_center);
