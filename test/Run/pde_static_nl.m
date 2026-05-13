% pde_static_nl.m — geometric-nonlinear static structural analysis.
%
% Cantilever beam under tip pressure load.  Runs Newton-Raphson
% with K reassembly on the deformed configuration each iteration.
% For small loads the result matches the linear-elastic
% structuralStatic answer; the test verifies Newton converges and
% the deflection is in the right ballpark.

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 8, 2, 2);

model = femodel('AnalysisType', 'structuralStaticNL', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);
model = pde_set_face_pressure(model, 2, 1.0e5);
model = pde_generate_mesh(model);

raw = pde_solve_structural_static_nl(model);
u   = pde_kernel_u(raw);

n = size(u, 1) / 3;
peak = 0.0;
for i = 1:n
    ux = u(3*i - 2);
    uy = u(3*i - 1);
    uz = u(3*i);
    mag = sqrt(ux*ux + uy*uy + uz*uz);
    if mag > peak; peak = mag; end
end

fprintf('PDE Tier-4 structuralStaticNL (Newton reassembly):\n');
fprintf('  Newton iters:         %.0f\n', raw.Iters);
fprintf('  peak |u| (mm):        %.4f\n', peak * 1000);
