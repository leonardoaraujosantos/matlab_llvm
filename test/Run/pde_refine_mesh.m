% pde_refine_mesh.m — uniform 2× cuboid refinement.
%
% Verifies that refineMesh on a 2×2×2 cuboid returns a 4×4×4 mesh
% with the same overall geometry (W/D/H) and that downstream FEM
% assembly still works correctly.  Also runs adaptmesh as a global
% refinement v1 (same as refineMesh for now).

gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 2, 1, 1);
gm2 = refineMesh(gm);
gm3 = adaptmesh(gm, 0.5);   % v1: same as refineMesh

n_in   = size(pde_mesh_nodes(gm),  1);
n_ref  = size(pde_mesh_nodes(gm2), 1);
n_adapt = size(pde_mesh_nodes(gm3), 1);

% Re-run a simple Poisson solve on the refined mesh to confirm it
% is a valid mesh.
model = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm2);
model = pde_set_material(model, ...
            materialProperties('ThermalConductivity', 50.0));
model = pde_set_face_temperature(model, 5, 100.0);
model = pde_set_face_temperature(model, 6,   0.0);
model = pde_generate_mesh(model);

R = pde_solve(model);
T = pde_kernel_u(R);

% Average temperature should be ~50 °C (linear gradient mean).
nn = size(T, 1);
sumT = 0.0;
for i = 1:nn
    sumT = sumT + T(i);
end
avgT = sumT / nn;

fprintf('PDE Tier-4 refineMesh + adaptmesh:\n');
fprintf('  input  nodes:         %.0f\n', n_in);
fprintf('  refined nodes:        %.0f\n', n_ref);
fprintf('  adaptmesh nodes:      %.0f\n', n_adapt);
fprintf('  refined avg T (C):    %.0f\n', round(avgT));
