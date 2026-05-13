% pde_static_tl.m — Total-Lagrangian Newton (geometric nonlinear).
%
% Same cantilever as pde_static_nl, solved via the TL path.  The
% TL kernel adds the geometric stiffness contribution K_geo
% on top of the K_mat reassembly; for moderate loads the answer
% converges to ~the linear-elastic result.  Output adds a
% ResNorm diagnostic (the final-displacement L2 norm) used by
% post-processors to track load-step progress.

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 8, 2, 2);

model = femodel('AnalysisType', 'structuralStaticTL', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);
model = pde_set_face_pressure(model, 2, 1.0e5);
model = pde_generate_mesh(model);

raw = pde_solve_structural_static_tl(model);
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

fprintf('PDE Tier-4 structuralStaticTL (Total-Lagrangian Newton):\n');
fprintf('  Newton iters:        %.0f\n', raw.Iters);
fprintf('  peak |u| (mm):       %.4f\n', peak * 1000);
fprintf('  log10(ResNorm):      %.0f\n', floor(log10(raw.ResNorm)));
