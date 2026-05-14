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
% looks blocky.  0.012 is the sweet spot: ~25 k nodes / ~75 k
% tets / ~75 k DOFs; PCG converges in a few minutes.  Drop to
% 0.008 for finer detail at the cost of ~10× compute.
voxel = 0.012;
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
res    = pcg(Kc, Fc, 1.0e-4, 30000.0);
u      = pcg_x(res);
flag   = pcg_flag(res);
iters  = pcg_iter(res);
relres = pcg_relres(res);

% --- Post-process: per-node vM + render --------------------------
vm_node = pde_node_von_mises_3d(mesh, u, E, nu);
disp    = pde_reshape_disp_3d(u);

% --- Wind direction arrow baked into the mesh --------------------
% Append a short arrow-shaped triangular ribbon on the windward
% side (-y direction at mid-height z=0).  The arrow is built from
% 5 extra nodes and 3 extra triangles; we give those node rows a
% high vM marker value so they render in the warm/red end of the
% colour map, visibly contrasted against the antenna body.
%
% Arrow geometry (in the FLIPPED frame, after the z-flip above):
%   tail   = ( 0,    -0.70, 0)
%   shaft  = ( 0,    -0.40, 0)   (where the head starts)
%   tip    = ( 0,    -0.32, 0)
%   leftV  = (-0.06, -0.42, 0)
%   rightV = ( 0.06, -0.42, 0)
%
% Triangles (1-based node refs into the extended Nodes array):
%   shaft strip (2 thin rectangles -> 4 triangles):
%     a thin rectangle around the shaft line, in the z=0 plane
%   tip:  shaft -> leftV -> rightV  (an arrowhead triangle)
%
% The antenna's existing surface nodes / faces are kept; we
% append the arrow vertices and faces.

n_orig = size(nodes, 1);
arr_pts = [
   0.00, -0.70, -0.02;
   0.00, -0.70,  0.02;
   0.00, -0.40, -0.02;
   0.00, -0.40,  0.02;
   0.00, -0.32,  0.00;
  -0.06, -0.42,  0.00;
   0.06, -0.42,  0.00
];

% Build extended Nodes ((Nn + 7) x 3)
ntot = n_orig + 7;
ext_nodes = zeros(ntot, 3);
for i = 1:n_orig
    ext_nodes(i, 1) = nodes(i, 1);
    ext_nodes(i, 2) = nodes(i, 2);
    ext_nodes(i, 3) = nodes(i, 3);
end
for i = 1:7
    ext_nodes(n_orig + i, 1) = arr_pts(i, 1);
    ext_nodes(n_orig + i, 2) = arr_pts(i, 2);
    ext_nodes(n_orig + i, 3) = arr_pts(i, 3);
end

% Build extended Faces.  Each face row is [face_id, n1, n2, n3].
% We use face_id = 99 for the arrow triangles so it doesn't
% collide with the mesh's 1..6 ids.
nf_orig = size(faces, 1);
ftot = nf_orig + 4;
ext_faces = zeros(ftot, 4);
for i = 1:nf_orig
    ext_faces(i, 1) = faces(i, 1);
    ext_faces(i, 2) = faces(i, 2);
    ext_faces(i, 3) = faces(i, 3);
    ext_faces(i, 4) = faces(i, 4);
end
% Shaft ribbon -- 2 triangles forming a thin rectangle
ext_faces(nf_orig + 1, 1) = 99;
ext_faces(nf_orig + 1, 2) = n_orig + 1;
ext_faces(nf_orig + 1, 3) = n_orig + 2;
ext_faces(nf_orig + 1, 4) = n_orig + 4;
ext_faces(nf_orig + 2, 1) = 99;
ext_faces(nf_orig + 2, 2) = n_orig + 1;
ext_faces(nf_orig + 2, 3) = n_orig + 4;
ext_faces(nf_orig + 2, 4) = n_orig + 3;
% Arrowhead -- 2 triangles forming the V
ext_faces(nf_orig + 3, 1) = 99;
ext_faces(nf_orig + 3, 2) = n_orig + 5;
ext_faces(nf_orig + 3, 3) = n_orig + 6;
ext_faces(nf_orig + 3, 4) = n_orig + 3;
ext_faces(nf_orig + 4, 1) = 99;
ext_faces(nf_orig + 4, 2) = n_orig + 5;
ext_faces(nf_orig + 4, 3) = n_orig + 4;
ext_faces(nf_orig + 4, 4) = n_orig + 7;

% Extended vm field: arrow nodes get the global peak vM as marker
% so they render in the bright end of the colour ramp.
peak_marker = 0.0;
for i = 1:n_orig
    v = vm_node(i);
    if v > peak_marker; peak_marker = v; end
end
ext_vm = zeros(ntot, 1);
for i = 1:n_orig
    ext_vm(i) = vm_node(i);
end
for i = 1:7
    ext_vm(n_orig + i) = peak_marker;
end

% Exaggerate displacement 500× on the antenna nodes (arrow stays
% put at its reference position).
ext_disp = zeros(ntot, 3);
for i = 1:n_orig
    ext_disp(i, 1) = disp(i, 1);
    ext_disp(i, 2) = disp(i, 2);
    ext_disp(i, 3) = disp(i, 3);
end

pdeplot3d_deform_scale(500.0);
pdeplot3d_deformation(ext_disp);
pdeplot3d(ext_nodes, ext_faces, ext_vm);

title('Antenna 5G: 200 km/h wind on -y face (arrow), von Mises (Pa)');
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
