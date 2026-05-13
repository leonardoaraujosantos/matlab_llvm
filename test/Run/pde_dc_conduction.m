% pde_dc_conduction.m — Tier-3 DC conduction via femodel.
%
% Solves Ohm's law -div(sigma grad V) = 0 on a copper bar:
%   V = 1 V on face 5 (the "input" terminal)
%   V = 0 V on face 6 (the "output" terminal)
% Midpoint V = 0.5 V (linear profile, σ-invariant).

gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 8, 2, 2);

model = femodel('AnalysisType', 'dcConduction', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('ElectricalConductivity', 5.96e7));  % copper
model = pde_set_face_voltage(model, 5, 1.0);
model = pde_set_face_voltage(model, 6, 0.0);
model = pde_generate_mesh(model);

raw = pde_solve(model);
V   = pde_kernel_u(raw);

% Midpoint at (0.05, 0.01, 0.01).
nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.05;
    dy = nodes(i, 2) - 0.01;
    dz = nodes(i, 3) - 0.01;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end
V_mid = V(best_i);

fprintf('PDE Tier-3 dcConduction: 0.1m copper bar, V=1|0 ends\n');
fprintf('  sigma:                5.96e7 S/m (copper)\n');
fprintf('  V(midpoint) = %.2f V\n', V_mid);
fprintf('  analytic    = 0.5 V\n');
