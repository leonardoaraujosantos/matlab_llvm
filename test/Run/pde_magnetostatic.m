% pde_magnetostatic.m — Tier-3 magnetostatic via femodel.
%
% Solves -div((1/mu) grad A) = 0 on a 1m x 0.2m x 0.2m bar with:
%   A = 10 on face 5 (x = 0)
%   A =  0 on face 6 (x = W)
% Analytic: A(midpoint) = 5.

gm = pde_mesh_cuboid_tet(1.0, 0.2, 0.2, 10, 2, 2);

model = femodel('AnalysisType', 'magnetostatic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('RelativePermeability', 1000.0));
model = pde_set_face_potential(model, 5, 10.0);
model = pde_set_face_potential(model, 6,  0.0);
model = pde_generate_mesh(model);

raw = pde_solve(model);
A   = pde_kernel_u(raw);

% Midpoint node at (0.5, 0.1, 0.1).
nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.5;
    dy = nodes(i, 2) - 0.1;
    dz = nodes(i, 3) - 0.1;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end
A_mid = A(best_i);

fprintf('PDE Tier-3 magnetostatic: 1m bar, A=10|0 ends\n');
fprintf('  mu_r:                 1000\n');
fprintf('  A(midpoint) = %.1f (rounded)\n', round(A_mid));
fprintf('  analytic    = 5.0\n');
