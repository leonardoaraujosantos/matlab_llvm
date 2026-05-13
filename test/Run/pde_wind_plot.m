% pde_wind_plot.m — Tier-2 headline demo + pdeplot3D rendering.
%
% Runs the wind-stress 3-D analysis and renders the von Mises stress
% map on the deformed boundary mesh to PNG.  This validates the
% pdeplot3D unstructured-mesh painter end-to-end.

% Wind load
rho_air = 1.225;
v_kmh   = 250;
v_ms    = v_kmh / 3.6;
q_dyn   = 0.5 * rho_air * v_ms * v_ms;
Cd      = 1.2;
p_wind  = Cd * q_dyn;

% Geometry: 3 m x 0.05 m x 2 m sign-panel.
mesh = pde_mesh_cuboid_tet(3.0, 0.05, 2.0, 12, 1, 8);

E  = 2.0e11;
nu = 0.30;
K = pde_assemble_elast_3d(mesh, E, nu);
F = pde_face_pressure_3d(mesh, 3.0, p_wind);
fixed_nodes = pde_face_nodes(mesh, 5.0);
sys2 = pde_apply_fixed_3d(K, F, fixed_nodes);
Kc = pde_sys_K(sys2);
Fc = pde_sys_F(sys2);
u  = Kc \ Fc;

% Per-node von Mises for Gouraud-shaded rendering.
vm_node = pde_node_von_mises_3d(mesh, u, E, nu);
disp    = pde_reshape_disp_3d(u);

nodes = pde_mesh_nodes(mesh);
faces = pde_mesh_faces(mesh);

% Render the stress map on the deformed shape.
pdeplot3d_deform_scale(100);
pdeplot3d_deformation(disp);
pdeplot3d(nodes, faces, vm_node);
title('Wind-stress demo: 250 km/h on a sign-panel');
saveas(gcf, '/tmp/pde_wind.png');

% Numeric summary so we have a deterministic stdout to gold-test.
def = pde_peak_disp_3d(u);

% Peak vM by linear scan over the node array.
n_nodes = size(vm_node, 1);
peak_vm = 0.0;
for i = 1:n_nodes
    v = vm_node(i);
    if v > peak_vm
        peak_vm = v;
    end
end

fprintf('PDE pdeplot3D: 250 km/h wind on sign-panel\n');
fprintf('  nodes / faces:        %.0f / %.0f\n', size(nodes, 1), size(faces, 1));
fprintf('  peak displacement:    %.3f mm\n', def * 1000);
fprintf('  peak nodal vM:        %.2f MPa\n', peak_vm / 1e6);
fprintf('  rendered:             /tmp/pde_wind.png\n');
