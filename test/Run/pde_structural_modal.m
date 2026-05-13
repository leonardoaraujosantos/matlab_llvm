% pde_structural_modal.m — Tier-3 structuralModal via femodel.
%
% Unconstrained modal analysis of a steel block: the first 6
% eigenvalues should be near-zero (rigid-body modes); the 7th and
% above are the physical flexible modes.

gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 5, 2, 2);

model = femodel('AnalysisType', 'structuralModal', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_num_modes(model, 10);
model = pde_generate_mesh(model);

raw = pde_solve(model);
freqs = pde_kernel_freqs(raw);
nf = size(freqs, 1);

% Categorize: rigid-body (freq < 1 Hz) vs flexible (freq > 1 Hz).
n_rigid = 0;
n_flex = 0;
first_flex_hz = 0.0;
for i = 1:nf
    f = freqs(i);
    if f < 1.0
        n_rigid = n_rigid + 1;
    else
        n_flex = n_flex + 1;
        if first_flex_hz == 0.0
            first_flex_hz = f;
        end
    end
end

fprintf('PDE Tier-3 structuralModal: 0.1m x 0.02m x 0.02m steel block\n');
fprintf('  modes requested:      10\n');
fprintf('  modes returned:       %.0f\n', nf);
fprintf('  rigid-body modes:     %.0f (expected ~6)\n', n_rigid);
fprintf('  flexible modes:       %.0f\n', n_flex);
fprintf('  first flex frequency: %.0f Hz (order of magnitude)\n', ...
        round(first_flex_hz / 1000) * 1000);
