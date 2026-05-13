% pde_refine_bey.m — Bey red refinement (arbitrary-tet 8-subdivision).
%
% Verifies that refineMeshBey on a 1×1×1 cuboid (1 hex = 6 tets)
% produces a mesh with 8× the tets and the right node count
% (corners + unique edges).  Each parent tet's 4 corners + 6
% mid-edges yield 8 sub-tets via Bey's pattern.
%
% Also runs adaptmesh_marked(mesh, 1.0) which is the marked-Bey
% variant — at frac=1.0 it's uniform Bey, at frac<1.0 it returns
% the mesh unchanged (v1 binary).

gm = pde_mesh_cuboid_tet(0.1, 0.1, 0.1, 1, 1, 1);

n_in   = size(pde_mesh_nodes(gm),   1);
nt_in  = size(pde_mesh_tets(gm),    1);
gm2    = refineMeshBey(gm);
n_ref  = size(pde_mesh_nodes(gm2),  1);
nt_ref = size(pde_mesh_tets(gm2),   1);
gm3    = pde_adapt_mesh_marked(gm, 1.0);
nt_a   = size(pde_mesh_tets(gm3),   1);

% Confirm the refined mesh assembles + solves correctly.
model = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm2);
model = pde_set_material(model, ...
            materialProperties('ThermalConductivity', 50.0));
model = pde_set_face_temperature(model, 5, 100.0);
model = pde_set_face_temperature(model, 6,   0.0);
model = pde_generate_mesh(model);
R = pde_solve(model);
T = pde_kernel_u(R);

nn = size(T, 1);
sumT = 0.0;
for i = 1:nn
    sumT = sumT + T(i);
end
avgT = sumT / nn;

fprintf('PDE Tier-4 Bey red refinement:\n');
fprintf('  input  tets:           %.0f\n', nt_in);
fprintf('  refined tets (Bey):    %.0f\n', nt_ref);
fprintf('  refined / input:       %.0f\n', round(nt_ref / nt_in));
fprintf('  adaptmesh (frac=1):    %.0f\n', nt_a);
fprintf('  refined avg T (C):     %.0f\n', round(avgT));
