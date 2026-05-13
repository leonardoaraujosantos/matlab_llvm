% pde_thermal_picard.m — nonconstant k(T) via Picard outer loop.
%
% Same 100 C / 0 C bar as the steady thermal test but with k(T) =
% k0 (1 + α_k T).  The model picks up the Picard branch
% automatically when materialProperties.ThermalCondCoeff is set.
% For α_k > 0 the effective conductivity scales up with mean T
% so the midpoint temperature drifts from the pure-linear 40 °C
% (snapped) profile.

gm = pde_mesh_cuboid_tet(0.1, 0.05, 0.05, 5, 2, 2);

model = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('ThermalConductivity', 50.0, ...
                               'ThermalCondCoeff',    0.01));
model = pde_set_face_temperature(model, 5, 100.0);
model = pde_set_face_temperature(model, 6,   0.0);
model = pde_generate_mesh(model);

R = pde_solve(model);
T = pde_kernel_u(R);

nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.05;
    dy = nodes(i, 2) - 0.025;
    dz = nodes(i, 3) - 0.025;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end

fprintf('PDE thermalSteady Picard k(T):\n');
fprintf('  alpha_k:              0.01\n');
fprintf('  T_mid (C, rounded):   %.0f\n', round(T(best_i)));
