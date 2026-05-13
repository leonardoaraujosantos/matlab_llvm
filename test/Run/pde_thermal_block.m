% pde_thermal_block.m — Tier-3 thermal steady-state via femodel.
%
% Solves -div(k grad T) = 0 on a 1m x 0.2m x 0.2m steel slab with
%   T = 100 C on face 5 (x = 0, the "hot" end)
%   T =   0 C on face 6 (x = W, the "cold" end)
% Other faces are adiabatic (default zero-Neumann).
%
% The analytical solution for this 1-D conduction problem is a linear
% profile from 100 to 0; we check the midpoint value at x = 0.5.

gm = pde_mesh_cuboid_tet(1.0, 0.2, 0.2, 10, 2, 2);

model = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('ThermalConductivity', 50.0));
model = pde_set_face_temperature(model, 5, 100.0);
model = pde_set_face_temperature(model, 6,   0.0);
model = pde_generate_mesh(model);

raw = pde_solve(model);
u   = pde_kernel_u(raw);

% Find the node at (0.5, 0.1, 0.1) — the midpoint of the slab.
nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.5;
    dy = nodes(i, 2) - 0.1;
    dz = nodes(i, 3) - 0.1;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end
T_mid = u(best_i);

fprintf('PDE Tier-3 thermal: 1m steel slab, T=100|0 ends\n');
fprintf('  mesh:                 10 x 2 x 2 hex (33 surface faces)\n');
fprintf('  k = 50 W/(m K)\n');
fprintf('  T(midpoint) = %.1f C (rounded)\n', round(T_mid));
fprintf('  analytic    = 50.0 C\n');
