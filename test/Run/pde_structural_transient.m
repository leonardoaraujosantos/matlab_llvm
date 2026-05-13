% pde_structural_transient.m — Tier-3 structuralTransient via Newmark.
%
% Cantilever beam released under a step pressure load.  Runs an
% explicit central-difference time-step loop; reports peak tip
% displacement reached during the simulation (which oscillates
% around the static deflection without damping).

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 10, 2, 2);

model = femodel('AnalysisType', 'structuralTransient', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);            % clamp x=0
model = pde_set_face_pressure(model, 2, 1.0e5);     % top-face pressure
model = pde_set_time_step(model, 2.0e-6);
model = pde_set_num_steps(model, 1000);
model = pde_generate_mesh(model);

raw   = pde_solve(model);
u_end = pde_kernel_u(raw);
Uh    = pde_kernel_uhist(raw);
tl    = pde_kernel_tlist(raw);

% Peak displacement magnitude across all nodes + all time steps.
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

fprintf('PDE Tier-3 structuralTransient: cantilever step-pressure\n');
fprintf('  dt = 2 us, steps = 1000, total t = 2 ms\n');
fprintf('  num time samples:    %.0f\n', size(tl, 1));
fprintf('  peak |u_i,k| (mm):   %.4g\n', peak * 1000);
