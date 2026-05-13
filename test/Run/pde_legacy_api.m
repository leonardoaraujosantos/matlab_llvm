% pde_legacy_api.m — MATLAB-faithful legacy entry-point names.
%
% Exercises the createpde-style surface: solvepde(model),
% solvepdeeig(model), specifyCoefficients(model, c, a, f),
% applyBoundaryCondition(model, face, val), pdegplot, pdemesh.
% Each is a thin wrapper that forwards to the existing kernel
% (pde_solve, etc.); the test verifies the names route correctly
% rather than re-validating the underlying physics.

% --- solvepde alias -------------------------------------------------
gm = pde_mesh_cuboid_tet(0.1, 0.05, 0.05, 5, 2, 2);
m1 = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm);
m1 = pde_set_material(m1, ...
        materialProperties('ThermalConductivity', 50.0));
m1 = applyBoundaryCondition(m1, 5, 100.0);   % Dirichlet 100°C at x=0
m1 = applyBoundaryCondition(m1, 6, 0.0);     %             0°C at x=L
m1 = pde_generate_mesh(m1);
R1 = solvepde(m1);
T1 = pde_kernel_u(R1);

nodes = pde_mesh_nodes(m1.Mesh);
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

% --- solvepdeeig alias ---------------------------------------------
gm2 = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 4, 2, 2);
m2 = femodel('AnalysisType', 'structuralModal', 'Geometry', gm2);
m2 = pde_set_material(m2, ...
        materialProperties('YoungsModulus', 2.0e11, ...
                           'PoissonsRatio', 0.30, ...
                           'MassDensity',   7850));
m2 = pde_set_num_modes(m2, 10);
m2 = pde_generate_mesh(m2);
R2 = solvepdeeig(m2);
freqs = pde_kernel_freqs(R2);

% --- specifyCoefficients (scalar PDE form) -------------------------
m3 = femodel('AnalysisType', 'electrostatic', 'Geometry', gm);
m3 = pde_set_material(m3, ...
        materialProperties('RelativePermittivity', 1.0));
m3 = specifyCoefficients(m3, 1.0, 0.0, 0.0);
m3 = applyBoundaryCondition(m3, 5, 5.0);
m3 = applyBoundaryCondition(m3, 6, 0.0);
m3 = pde_generate_mesh(m3);
R3 = solvepde(m3);
V3 = pde_kernel_u(R3);

fprintf('PDE legacy API (solvepde / solvepdeeig / specifyCoefficients):\n');
fprintf('  thermal T_mid (C):    %.0f\n', round(T1(best_i)));
fprintf('  modal n freqs:        %.0f\n', size(freqs, 1));
fprintf('  electrostatic V_mid:  %.1f\n', V3(best_i));
