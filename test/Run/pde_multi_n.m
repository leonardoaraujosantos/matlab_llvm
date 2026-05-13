% pde_multi_n.m — N-component coupled scalar Poisson (N = 3).
%
% Generalises pde_multi_component to arbitrary N components.
% Inputs are vectors / matrices on the model:
%   .MultiCN  (N × 1)  diffusion c_i
%   .MultiAN  (N × N)  reaction matrix a_ij
%   .MultiFN  (N × 1)  body source f_i

gm = pde_mesh_cuboid_tet(0.1, 0.05, 0.05, 3, 2, 2);
model = femodel('Geometry', gm);

% N = 3 decoupled Poisson systems with f1=1, f2=2, f3=3.
c = zeros(3, 1);
c(1) = 1.0; c(2) = 1.0; c(3) = 1.0;
A = zeros(3, 3);
A(1, 1) = 1.0;
A(2, 2) = 1.0;
A(3, 3) = 1.0;
f = zeros(3, 1);
f(1) = 1.0; f(2) = 2.0; f(3) = 3.0;
model = pde_set_multi_coeff_n(model, c, A, f);
model = pde_generate_mesh(model);

R = pde_solve_multi_n(model);
u1 = pde_multi_n_u(R, 1);
u2 = pde_multi_n_u(R, 2);
u3 = pde_multi_n_u(R, 3);

nn = size(u1, 1);
s1 = 0; s2 = 0; s3 = 0;
for i = 1:nn
    s1 = s1 + u1(i);
    s2 = s2 + u2(i);
    s3 = s3 + u3(i);
end
avg1 = s1 / nn;
avg2 = s2 / nn;
avg3 = s3 / nn;

fprintf('PDE Tier-4 multi-component (N=3 coupled Poisson):\n');
fprintf('  N:                 %.0f\n', R.N);
fprintf('  num nodes:         %.0f\n', nn);
fprintf('  u2/u1 ratio:       %.0f\n', round(avg2 / avg1));
fprintf('  u3/u1 ratio:       %.0f\n', round(avg3 / avg1));
