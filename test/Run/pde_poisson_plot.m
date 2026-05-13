% pde_poisson_plot.m — Tier-1 + pdeplot 2-D rendering smoke test.
%
% Solves -Lap(u) = 1 on [0,1]^2 sparsely + renders the solution
% with pdeplot.  PNG written to /tmp/pde_poisson.png.

mesh = pde_mesh_rect_tri(0.0, 1.0, 0.0, 1.0, 21, 21);
bnd  = pde_boundary_nodes_rect(mesh);
sys  = pde_assemble_poisson_2d_sparse(mesh, 1.0, 0.0, 1.0);
sys2 = pde_apply_dirichlet_sparse(sys, bnd, 0.0);
K    = pde_sys_K_sparse(sys2);
F    = pde_sys_F(sys2);
res  = pcg(K, F, 1.0e-10, 500.0);
u    = pcg_x(res);

% Render
nodes = pde_mesh_nodes(mesh);
tris  = pde_mesh_triangles(mesh);
pdeplot(nodes, tris, u);
title('Poisson: -Lap(u) = 1, u = 0 on boundary');
xlabel('x');
ylabel('y');
saveas(gcf, '/tmp/pde_poisson.png');

% Centre value as the gold signal
u_center = u(221);

fprintf('PDE 2-D pdeplot: -Lap(u) = 1 on [0,1]^2\n');
fprintf('  nodes:                %.0f\n', size(nodes, 1));
fprintf('  triangles:            %.0f\n', size(tris, 1));
fprintf('  PCG iters:            %.0f\n', pcg_iter(res));
fprintf('  u(0.5, 0.5):          %.4f\n', u_center);
fprintf('  rendered:             /tmp/pde_poisson.png\n');
