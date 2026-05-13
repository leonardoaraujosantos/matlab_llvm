% pde_clamped_plate.m — Tier-2 3D linear-elasticity FEM smoke test.
%
% A 0.5 m x 0.05 m x 0.05 m steel cantilever beam is fixed on its
% left face (x = 0, face 5) and loaded with a uniform 100 kPa pressure
% on its top face (z = H, face 2).
%
% Validates the multicuboid -> assemble_elast_3d -> face_pressure ->
% apply_fixed -> mldivide -> von_mises pipeline.  See
% docs/pde_toolbox_roadmap.md section 3.

W = 0.5; D = 0.05; H = 0.05;
Nx = 10; Ny = 2; Nz = 2;
mesh = pde_mesh_cuboid_tet(W, D, H, Nx, Ny, Nz);

% Structural steel
E  = 2.0e11;     % Young's modulus, Pa
nu = 0.30;       % Poisson's ratio

K = pde_assemble_elast_3d(mesh, E, nu);

% Top-face uniform pressure 100 kPa.  Face 2 = z = H plane.
F = pde_face_pressure_3d(mesh, 2.0, 1.0e5);

% Fixed-base boundary: clamped on face 5 (x = 0, the cantilever root).
fixed_nodes = pde_face_nodes(mesh, 5.0);
sys2 = pde_apply_fixed_3d(K, F, fixed_nodes);

Kc = pde_sys_K(sys2);
Fc = pde_sys_F(sys2);
u  = Kc \ Fc;

vm  = pde_von_mises_3d(mesh, u, E, nu);
def = pde_peak_disp_3d(u);

% Round to a single digit so floating-point jitter doesn't fail the
% gold compare.
def_um = round(def * 1e6);     % microns

fprintf('PDE Tier-2: cantilever 0.5m x 50mm x 50mm steel beam\n');
fprintf('  mesh:                 10 x 2 x 2 hex (99 nodes, 240 tets)\n');
fprintf('  E = 200 GPa, nu = 0.3\n');
fprintf('  load:                 100 kPa pressure on top face\n');
fprintf('  fixed dofs:           %.0f nodes on root face\n', size(fixed_nodes, 1));
fprintf('  peak displacement:    %.0f microns\n', def_um);
fprintf('  K rows:               %.0f\n', size(K, 1));
fprintf('  vM count:             %.0f\n', size(vm, 1));
