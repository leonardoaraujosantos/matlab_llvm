% pde_quad_tet_stress.m — T10 per-node von Mises stress recovery.
%
% Same uniaxial-pull problem as pde_quad_tet: σ = 1 MPa applied
% on the +x face of a clamped 0.5 m × 50 mm × 50 mm steel bar.
% The exact stress field is σ_xx = 1 MPa uniform; recovery should
% match to within numerical roundoff.

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 4, 1, 1);

E  = 2.0e11;
nu = 0.30;
p  = 1.0e6;

mq    = pde_mesh_quadratic(gm);
K_sp  = pde_assemble_elast_3d_t10(mq, E, nu);
F     = pde_face_pressure_3d_t10(mq, 6, -p);
fixed = pde_face_nodes_t10(mq, 5);
sys2  = pde_apply_fixed_3d_t10(K_sp, F, fixed);
gr    = sparse_gmres_ilu0(pde_sys_K(sys2), pde_sys_F(sys2), 1e-10, 4000);
u10   = pcg_x(gr);

vm = pde_node_von_mises_3d_t10(mq, u10, E, nu);

% Average of vm over interior nodes (skip clamped face = 0 stress)
nn = size(vm, 1);
sum_vm = 0.0;
cnt = 0;
for i = 1:nn
    v = vm(i);
    if v > 1e3   % filter out near-zero nodes (clamped face)
        sum_vm = sum_vm + v;
        cnt = cnt + 1;
    end
end
avg_vm = sum_vm / cnt;

fprintf('PDE T10 stress recovery (uniaxial pull):\n');
fprintf('  T10 mesh nodes:        %.0f\n', nn);
fprintf('  avg von Mises (MPa):   %.3f\n', avg_vm / 1.0e6);
fprintf('  analytic sigma_VM:     %.3f MPa\n', 1.0);
