% pde_rom.m — modal-truncation Reduced-Order Model (ROM).
%
% Build a 12-mode reduced model of a clamped steel cantilever,
% then reconstruct the displacement field from a unit modal
% excitation.  Verifies that:
%   1) reduce(model) returns a valid R (3N × n_modes) matrix.
%   2) reconstructSolution(Rred, q) maps modal → physical correctly.

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 8, 2, 2);

model = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed(model, 5);
model = pde_set_num_modes(model, 12);
model = pde_generate_mesh(model);

Rred = reduce(model);
nm   = Rred.nModes;

% Excite the first physical mode (after rigid-body / spurious filtering).
q = zeros(nm, 1);
q(1) = 1.0;
u = reconstructSolution(Rred, q);

% Peak |u| over all DOFs.
nd = size(u, 1);
peak = 0.0;
for i = 1:nd
    v = u(i);
    if v < 0; v = -v; end
    if v > peak; peak = v; end
end

fprintf('PDE Tier-4 ROM (modal truncation):\n');
fprintf('  reduced modes:    %.0f\n', nm);
fprintf('  num DOFs:         %.0f\n', Rred.NumDOFs);
fprintf('  log10(peak|u|):   %.0f\n', floor(log10(peak)));
