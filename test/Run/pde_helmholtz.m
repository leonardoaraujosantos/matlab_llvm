% pde_helmholtz.m — harmonicElectromagnetic scalar wave equation.
%
% Solves -∇·(c∇u) - k²u = 0 inside a cuboid with V=1 on one face,
% V=0 on the opposite face.  The k=0 case reduces to electrostatic
% (linear potential); for k > 0 the solution oscillates.

% Nx=10 so a node lands exactly at x = 0.05 (the midpoint).
gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 10, 2, 2);

model = femodel('AnalysisType', 'harmonicElectromagnetic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('RelativePermittivity', 1.0, ...
                               'RelativePermeability', 1.0));
% Small wave number — Helmholtz is barely indefinite, behaves close
% to electrostatic.  Larger k would test resonance behavior.
model = pde_set_wave_number(model, 1.0);
model = pde_set_face_voltage(model, 5, 1.0);
model = pde_set_face_voltage(model, 6, 0.0);
model = pde_generate_mesh(model);

raw = pde_solve(model);
u   = pde_kernel_u(raw);

% Midpoint near (0.05, 0.01, 0.01).
nodes = pde_mesh_nodes(model.Mesh);
n = size(nodes, 1);
best_i = 1;
best_d = 1e9;
for i = 1:n
    dx = nodes(i, 1) - 0.05;
    dy = nodes(i, 2) - 0.01;
    dz = nodes(i, 3) - 0.01;
    d = dx * dx + dy * dy + dz * dz;
    if d < best_d
        best_d = d;
        best_i = i;
    end
end
u_mid = u(best_i);

fprintf('PDE Helmholtz (harmonicElectromagnetic):\n');
fprintf('  geom:                 0.1 x 0.02 x 0.02 m\n');
fprintf('  k = 1 rad/m, eps_r = 1, mu_r = 1\n');
fprintf('  u_top / u_bottom:     1 / 0\n');
fprintf('  u(midpoint) = %.2f (rounded)\n', round(u_mid * 100) / 100);
fprintf('  ~analytic   = 0.5 (low-k limit)\n');
