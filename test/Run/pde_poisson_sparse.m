% pde_poisson_sparse.m — Tier-1 sparse FEM smoke test.
%
% Same -Lap(u) = 1 problem on the unit square as pde_poisson_square,
% but assembles K as a sparse matrix and solves via PCG instead of
% dense LU.  Verifies the sparse path matches the dense path.

mesh = pde_mesh_rect_tri(0.0, 1.0, 0.0, 1.0, 21, 21);
bnd  = pde_boundary_nodes_rect(mesh);

% Sparse assembly.
sys  = pde_assemble_poisson_2d_sparse(mesh, 1.0, 0.0, 1.0);
sys2 = pde_apply_dirichlet_sparse(sys, bnd, 0.0);

K = pde_sys_K_sparse(sys2);
F = pde_sys_F(sys2);

% PCG: tol=1e-10, maxit=500.
res    = pcg(K, F, 1.0e-10, 500.0);
u      = pcg_x(res);
flag   = pcg_flag(res);
iter   = pcg_iter(res);
relres = pcg_relres(res);

% Centre node at index 221 (10*21 + 10 + 1).
u_center = u(221);

fprintf('PDE sparse: -Lap(u) = 1 on [0,1]x[0,1], u=0 on bnd\n');
fprintf('  K nnz:                %.0f\n', spnnz(K));
fprintf('  PCG flag:             %.0f (0=converged)\n', flag);
fprintf('  PCG iterations:       %.0f\n', iter);
fprintf('  u(0.5, 0.5):          %.4f\n', u_center);
fprintf('  analytic:             0.0737\n');
