% pde_modal_transient.m — modal-superposition transient with Rayleigh
% damping.
%
% Same cantilever as pde_structural_transient, but the time
% integration runs on the 12-mode subspace from Lanczos shift-invert.
% Rayleigh damping α = 0, β = 1e-5 — light material damping so the
% step response converges to ~the static deflection instead of
% oscillating indefinitely.

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 8, 2, 2);

model = femodel('AnalysisType', 'structuralTransientModal', ...
                'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);            % clamp x = 0
model = pde_set_face_pressure(model, 2, 1.0e5);     % top-face step
model = pde_set_time_step(model, 2.0e-6);
model = pde_set_num_steps(model, 1000);
model = pde_set_num_modes(model, 12);
model = pde_set_rayleigh(model, 0.0, 1.0e-5);       % α, β
model = pde_generate_mesh(model);

raw = pde_solve_structural_transient_modal(model);
Uh  = pde_kernel_uhist(raw);
tl  = pde_kernel_tlist(raw);

ndof  = size(Uh, 1);
nstep = size(Uh, 2);
peak = 0.0;
for k = 1:nstep
    for i = 1:ndof
        v = Uh(i, k);
        if v < 0
            v = -v;
        end
        if v > peak
            peak = v;
        end
    end
end

fprintf('PDE structuralTransientModal (Rayleigh, 12 modes):\n');
fprintf('  num time samples:    %.0f\n', size(tl, 1));
fprintf('  peak |u| log10(mm):  %.0f\n', floor(log10(peak * 1000)));
