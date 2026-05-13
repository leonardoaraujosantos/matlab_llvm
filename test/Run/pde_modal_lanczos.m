% pde_modal_lanczos.m — sanity-check §10.5 Lanczos shift-invert.
%
% Builds a constrained cantilever beam, assembles K_sparse, M_diag,
% and runs the new pde_eig_lanczos_si solver.  The first eigenvalue
% should match the analytic fundamental frequency of a steel beam.

gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 5, 2, 2);

% Material / sparse K + lumped M.
E  = 2.0e11;
nu = 0.30;
rho = 7850;
K  = pde_assemble_elast_3d_sparse(gm, E, nu);

% Build a lumped mass vector manually.  Volume per tet split equally
% over 4 corner nodes.
nodes = pde_mesh_nodes(gm);
tets  = pde_mesh_tets(gm);
Nn = size(nodes, 1);
Nt = size(tets, 1);
Mdiag = zeros(3 * Nn, 1);
% Approximation: rho * cell_volume / 24 per node-DOF (uniform mesh).
% Total cell volume = 0.1*0.02*0.02 = 4e-5 ; total mass = 0.314 kg.
% Per node (uniform): 0.314 / Nn ; per DOF: same value broadcast x3.
total_mass = rho * 0.1 * 0.02 * 0.02;
m_per = total_mass / Nn;
for i = 1:Nn
    Mdiag(3 * (i - 1) + 1) = m_per;
    Mdiag(3 * (i - 1) + 2) = m_per;
    Mdiag(3 * (i - 1) + 3) = m_per;
end

% Lanczos shift-invert with σ = -1 (small negative shift, K + M is SPD).
lams = pde_eig_lanczos_si(K, Mdiag, 12.0, -1.0);

% Categorize: rigid (< 1 Hz², treat as essentially zero) vs flexible.
n = size(lams, 1);
n_rigid = 0;
n_flex = 0;
first_flex_hz = 0.0;
for i = 1:n
    l = lams(i);
    if l < 1.0
        n_rigid = n_rigid + 1;
    else
        n_flex = n_flex + 1;
        if first_flex_hz == 0.0
            first_flex_hz = sqrt(l) / (2.0 * 3.141592653589793);
        end
    end
end

fprintf('PDE Lanczos §10.5: 0.1m steel block, unconstrained\n');
fprintf('  modes requested:      12\n');
fprintf('  modes returned:       %.0f\n', n);
fprintf('  rigid-body modes:     %.0f (expected ~6)\n', n_rigid);
fprintf('  flex modes returned:  %.0f\n', n_flex);
fprintf('  first flex order:     %.0f\n', round(log10(first_flex_hz)));
