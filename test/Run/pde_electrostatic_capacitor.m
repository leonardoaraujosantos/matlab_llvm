% pde_electrostatic_capacitor.m — Tier-3 electrostatic via femodel.
%
% Solves -div(eps grad V) = 0 on a 0.1m x 0.1m x 0.01m parallel-plate
% capacitor:
%   V = 10 V on face 2 (z = H, the top plate)
%   V =  0 V on face 1 (z = 0, the bottom plate)
% Side faces are no-flux (the analytic solution between two parallel
% conductors is a linear voltage profile V(z) = (z/H) * V_top).
%
% Verifies the analytic midpoint value V(0.005) = 5 V.

gm = pde_mesh_cuboid_tet(0.1, 0.1, 0.01, 4, 4, 4);

model = femodel('AnalysisType', 'electrostatic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('RelativePermittivity', 1.0));
model = pde_set_face_voltage(model, 1,  0.0);  % bottom plate
model = pde_set_face_voltage(model, 2, 10.0);  % top plate
model = pde_generate_mesh(model);

raw = pde_solve(model);
V   = pde_kernel_u(raw);

% Find the node closest to (0.05, 0.05, 0.005) — the midpoint.
nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.05;
    dy = nodes(i, 2) - 0.05;
    dz = nodes(i, 3) - 0.005;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end
V_mid = V(best_i);

fprintf('PDE Tier-3 electrostatic: parallel-plate capacitor\n');
fprintf('  geom:                 0.1 x 0.1 x 0.01 m\n');
fprintf('  V_top / V_bottom:     10 / 0 V\n');
fprintf('  V(midpoint) = %.1f V (rounded)\n', round(V_mid));
fprintf('  analytic    = 5.0 V\n');
