% antenna_wind_stress.m — 200 km/h wind on a real 5 G antenna model.
%
% Loads the user-supplied antenna_5g.glb (5 G antenna, ~14 k surface
% vertices, ~10 k triangles), voxelizes it into a volumetric tet
% mesh, applies a 200 km/h horizontal wind pressure on the windward
% face, clamps the base, solves linear-elastic equilibrium, and
% renders the von Mises stress map to a PNG.
%
% Wind physics (sea-level standard atmosphere):
%   rho_air = 1.225 kg/m^3
%   v       = 200 km/h = 55.556 m/s
%   q_dyn   = 0.5 * rho * v^2 = 1890 Pa
%   Cd      = 1.0           (cylindrical / mixed body, conservative)
%   p_wind  = Cd * q_dyn = 1890 Pa
%
% Pipeline:
%   GLB import  -> surface mesh
%   voxelize    -> volumetric tet mesh (AABB / ray-cast inside test)
%   sparse 3-D elasticity assembly
%   Dirichlet clamp on the base
%   horizontal wind pressure on the windward face
%   ILU(0)-preconditioned GMRES solve
%   per-node vM stress recovery
%   pdeplot3D render with deformation scale -> /tmp/antenna_wind.png

% --- Wind physics --------------------------------------------------
rho_air = 1.225;
v_kmh   = 200;
v_ms    = v_kmh / 3.6;
q_dyn   = 0.5 * rho_air * v_ms * v_ms;
Cd      = 1.0;
p_wind  = Cd * q_dyn;

fprintf('Wind:   %.0f km/h, dynamic pressure %.0f Pa, p_wind %.0f Pa\n', ...
        v_kmh, q_dyn, p_wind);

% --- Load + flip + voxelize ---------------------------------------
surface = pde_load_glb("/tmp/antenna_5g.glb");
N_surf = pde_mesh_nodes(surface);
n_surf = size(N_surf, 1);
fprintf('GLB:  %.0f surface nodes, %.0f triangles\n', ...
        n_surf, size(pde_mesh_faces(surface), 1));

% The GLB has the antenna oriented with the radio housing at +z
% and the mast extending to -z, which renders "upside down"
% relative to the typical install orientation (housing at the
% BOTTOM, mast extending UP to attach to the support bracket).
% Flip the z-axis of the surface nodes so the natural install
% pose is what we render.  We re-use the SAME node array
% (pde_mesh_nodes returns the live storage), so the voxelizer
% picks up the flipped coords.
for i = 1:n_surf
    N_surf(i, 3) = -N_surf(i, 3);
end

% Voxel size — the antenna has thin members; voxel=0.05 is too
% coarse (mast falls through), 0.025 captures the topology but
% the slender mast still looks blocky.  0.015 gives a clean
% mast plus visible features on the housing.
voxel = 0.015;
mesh = pde_voxelize_surface(surface, voxel);
nodes = pde_mesh_nodes(mesh);
tets  = pde_mesh_tets(mesh);
faces = pde_mesh_faces(mesh);
fprintf('Vol:  %.0f nodes, %.0f tets, %.0f boundary faces (voxel=%.3f)\n', ...
        size(nodes, 1), size(tets, 1), size(faces, 1), voxel);

% --- Linear elasticity assembly ----------------------------------
% Aluminium 6061-T6 (common antenna housing material):
%   E  = 6.9e10 Pa, nu = 0.33, yield = 276 MPa.
E  = 6.9e10;
nu = 0.33;

K = pde_assemble_elast_3d_sparse(mesh, E, nu);

% --- Wind load + clamp -------------------------------------------
% Voxelizer face_id convention: 1=-z, 2=+z, 3=-y, 4=+y, 5=-x, 6=+x.
% Antenna's natural orientation has the long axis along z; wind
% blows in +y so it hits face 3 (-y).  Clamp the base on face 1.
F = pde_face_pressure_3d(mesh, 3.0, p_wind);
fixed_nodes = pde_face_nodes(mesh, 1.0);
sys2 = pde_apply_fixed_3d_sparse(K, F, fixed_nodes);

Kc = pde_sys_K_sparse(sys2);
Fc = pde_sys_F(sys2);

% --- Sparse PCG solve --------------------------------------------
% Linear elasticity K is SPD → PCG is faster than ILU(0)+GMRES.
% Loose tolerance (1e-4) because the slender mast pushes the
% condition number high enough that converging to 1e-6 needs
% ~50 k iterations.  1e-4 is fine for stress-visualisation
% accuracy.
res    = pcg(Kc, Fc, 1.0e-4, 20000.0);
u      = pcg_x(res);
flag   = pcg_flag(res);
iters  = pcg_iter(res);
relres = pcg_relres(res);

% --- Post-process: per-node vM + render --------------------------
vm_node = pde_node_von_mises_3d(mesh, u, E, nu);
disp    = pde_reshape_disp_3d(u);

% Exaggerate displacement 50× so the deformed-shape effect is
% visible at this load magnitude (typical for cantilever bending).
pdeplot3d_deform_scale(50.0);
pdeplot3d_deformation(disp);
pdeplot3d(nodes, faces, vm_node);
title('Antenna 5G: 200 km/h wind, von Mises stress (Pa)');
saveas(gcf, '/tmp/antenna_wind.png');

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

fprintf('\nFEM solve:\n');
fprintf('  K rows:                  %.0f\n', sprows(K));
fprintf('  K nnz:                   %.0f\n', spnnz(K));
fprintf('  fixed-base nodes:        %.0f\n', size(fixed_nodes, 1));
fprintf('  GMRES flag:              %.0f (0=converged)\n', flag);
fprintf('  GMRES iterations:        %.0f\n', iters);
fprintf('  GMRES relres:            %.2e\n', relres);
fprintf('  peak displacement:       %.3f mm\n', def * 1000);
fprintf('  peak nodal vM stress:    %.3f MPa\n', peak_vm / 1e6);
fprintf('  Al 6061-T6 yield:        276 MPa\n');
fprintf('  safety factor:           %.1f\n', 276e6 / peak_vm);
fprintf('  rendered:                /tmp/antenna_wind.png\n');
