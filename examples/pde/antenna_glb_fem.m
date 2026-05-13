% antenna_glb_fem.m — End-to-end FEM analysis of a real-world 3-D
% model.  Loads /tmp/antenna.glb (5G antenna, 14k surface vertices,
% 10k triangles), voxelizes the AABB into a tet mesh, applies a
% gravity-style body-equivalent face load, solves elasticity with
% sparse + PCG, and renders the von Mises stress map.
%
% Validates the full pipeline:
%   GLB import -> surface mesh
%     -> voxelize -> volumetric tet mesh
%     -> sparse 3-D elasticity assembly
%     -> Dirichlet (clamp the base)
%     -> face pressure (uniform on the top)
%     -> PCG solve
%     -> per-node vM
%     -> pdeplot3D render to PNG

% --- Load + voxelize ----------------------------------------------
surface = pde_load_glb("/tmp/antenna.glb");
fprintf('GLB:  %.0f surface nodes, %.0f triangles\n', ...
        size(pde_mesh_nodes(surface), 1), ...
        size(pde_mesh_faces(surface), 1));

% Voxel size — the antenna's AABB is roughly 0.73 x 0.60 x 1.91
% (the GLB ships in normalized units, longest axis is z).  voxel
% size 0.05 gives ~14x12x38 ≈ 6.4k cells; inside cells produce a
% few thousand tets after the ray-cast inside test.
mesh = pde_voxelize_surface(surface, 0.05);
nodes = pde_mesh_nodes(mesh);
tets  = pde_mesh_tets(mesh);
faces = pde_mesh_faces(mesh);
fprintf('Vol:  %.0f nodes, %.0f tets, %.0f boundary faces\n', ...
        size(nodes, 1), size(tets, 1), size(faces, 1));

% --- FEM: cantilever bending under uniform top-face pressure -----
E  = 7.0e10;     % aluminium-ish, Pa
nu = 0.33;
K = pde_assemble_elast_3d_sparse(mesh, E, nu);

% Uniform pressure on the top face (face_id 2 = +z by voxelizer
% convention) — simulates a load pushing the antenna down.
F = pde_face_pressure_3d(mesh, 2.0, 1.0e4);

% Clamp the bottom face (face_id 1 = -z) — antenna base.
fixed_nodes = pde_face_nodes(mesh, 1.0);
sys2 = pde_apply_fixed_3d_sparse(K, F, fixed_nodes);

Kc = pde_sys_K_sparse(sys2);
Fc = pde_sys_F(sys2);

% --- Sparse PCG solve --------------------------------------------
res    = pcg(Kc, Fc, 1.0e-6, 2000.0);
u      = pcg_x(res);
flag   = pcg_flag(res);
iters  = pcg_iter(res);
relres = pcg_relres(res);

% --- Post-process: vM + render -----------------------------------
vm_node = pde_node_von_mises_3d(mesh, u, E, nu);
disp    = pde_reshape_disp_3d(u);
pdeplot3d_deform_scale(50.0);
pdeplot3d_deformation(disp);
pdeplot3d(nodes, faces, vm_node);
title('Antenna 5G: voxelized FEM, vM stress under 10 kPa top load');
saveas(gcf, '/tmp/antenna_fem.png');

% --- Summary -----------------------------------------------------
def = pde_peak_disp_3d(u);
n_nodes = size(nodes, 1);
peak_vm = 0.0;
for i = 1:n_nodes
    v = vm_node(i);
    if v > peak_vm
        peak_vm = v;
    end
end

fprintf('\n');
fprintf('FEM solve:\n');
fprintf('  K rows:               %.0f\n', sprows(K));
fprintf('  K nnz:                %.0f\n', spnnz(K));
fprintf('  fixed dofs:           %.0f nodes\n', size(fixed_nodes, 1));
fprintf('  PCG flag:             %.0f (0=converged)\n', flag);
fprintf('  PCG iterations:       %.0f\n', iters);
fprintf('  PCG relres:           %.2e\n', relres);
fprintf('  peak displacement:    %.3f um\n', def * 1e6);
fprintf('  peak nodal vM:        %.3f MPa\n', peak_vm / 1e6);
fprintf('  rendered:             /tmp/antenna_fem.png\n');
