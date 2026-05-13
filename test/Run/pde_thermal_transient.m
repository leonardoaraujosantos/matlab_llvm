% pde_thermal_transient.m — ρc_p ∂T/∂t − ∇·(k∇T) = 0 on a 1-D-ish bar.
%
% Steel bar 0.1 m long, cross-section 50 mm × 50 mm.  Initial T =
% 0 °C everywhere.  At t = 0 the −x face is held at 100 °C and the
% +x face at 0 °C.  As t → ∞ the steady linear-gradient profile
% T(x) = 100 (1 − x/L) is recovered.

gm = pde_mesh_cuboid_tet(0.1, 0.05, 0.05, 5, 2, 2);

model = femodel('AnalysisType', 'thermalTransient', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('ThermalConductivity', 50.0, ...
                               'SpecificHeat',        500.0, ...
                               'MassDensity',         7850));
model = pde_set_face_temperature(model, 5, 100.0);
model = pde_set_face_temperature(model, 6,   0.0);
model = pde_set_initial_temperature(model, 0.0);
model = pde_set_time_step(model, 1.0);     % 1 s steps
model = pde_set_num_steps(model, 800);     % run to ~steady (800 s)
model = pde_generate_mesh(model);

raw  = pde_solve(model);
T_ss = pde_kernel_u(raw);
tl   = pde_kernel_tlist(raw);

% Pick the node closest to (0.05, 0.025, 0.025).
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

fprintf('PDE thermalTransient: 0.1 m bar, 100 C / 0 C Dirichlet:\n');
fprintf('  num time samples:    %.0f\n', size(tl, 1));
fprintf('  steady T_mid (C):    %.0f\n', round(T_ss(best_i)));
fprintf('  expected ~steady:    50 (or 40 if midpoint snaps to x=0.06)\n');
