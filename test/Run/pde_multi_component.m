% pde_multi_component.m — 2-component coupled scalar Poisson system.
%
% Solves
%   -∇·(c1 ∇u) + a11 u + a12 v = f1
%   -∇·(c2 ∇v) + a21 u + a22 v = f2
% on a cuboid with c1 = c2 = 1, a11 = a22 = 1, a12 = a21 = 0,
% f1 = 1, f2 = 2.  With decoupled (a12 = a21 = 0) the two
% solutions are independent scaled Poisson responses.

gm = pde_mesh_cuboid_tet(0.1, 0.05, 0.05, 3, 2, 2);

model = femodel('Geometry', gm);
model = pde_set_multi_coeff(model, ...
                            1.0, 1.0, 1.0, ...   % c1, a11, f1
                            1.0, 1.0, 2.0, ...   % c2, a22, f2
                            0.0, 0.0);           % a12, a21 (decoupled)
model = pde_generate_mesh(model);

R = pde_solve_multi(model);
u = pde_multi_u(R);
v = pde_multi_v(R);

nn = size(u, 1);
sum_u = 0.0;
sum_v = 0.0;
for i = 1:nn
    sum_u = sum_u + u(i);
    sum_v = sum_v + v(i);
end
avg_u = sum_u / nn;
avg_v = sum_v / nn;

fprintf('PDE Tier-4 multi-component (2-coupled Poisson):\n');
fprintf('  num nodes:         %.0f\n', nn);
fprintf('  log10(avg u):      %.0f\n', floor(log10(avg_u)));
fprintf('  log10(avg v):      %.0f\n', floor(log10(avg_v)));
fprintf('  v/u ratio:         %.0f\n', round(avg_v / avg_u));
