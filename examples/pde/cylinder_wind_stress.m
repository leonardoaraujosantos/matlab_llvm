% cylinder_wind_stress.m — Cantilever cylinder under 200 km/h wind.
%
% Reference sanity-check for the antenna_wind_stress demo: a simple
% solid aluminium cylinder of the same height as the antenna model
% (1.91 m), clamped at the base, loaded by horizontal wind.  The
% analytic cantilever-bending answer is closed-form so we can verify
% the FEM solution against it.
%
% Geometry:
%   R       = 0.05 m   (50 mm radius — typical antenna mast)
%   H       = 1.91 m   (same as the 5G antenna model's z-extent)
%   E       = 6.9e10 Pa, nu = 0.33 (aluminium 6061-T6)
%
% Analytic cantilever (uniform distributed lateral load):
%   w       = q * 2R * Cd        load per unit length (N/m)
%   M_max   = w * L^2 / 2        bending moment at the clamped base
%   I       = pi * R^4 / 4       second moment of area
%   sigma   = M_max * R / I      max bending stress (tension/compression
%                                 on the windward/leeward fibres)
%   delta   = w * L^4 / (8 E I)  tip deflection

% --- Wind physics --------------------------------------------------
rho_air = 1.225;
v_kmh   = 200;
v_ms    = v_kmh / 3.6;
q_dyn   = 0.5 * rho_air * v_ms * v_ms;
Cd      = 1.0;
p_wind  = Cd * q_dyn;

% --- Geometry + voxelize -------------------------------------------
R     = 0.05;
H     = 1.91;
voxel = 0.015;

mesh  = pde_multicylinder(R, H, voxel);
nodes = pde_mesh_nodes(mesh);
tets  = pde_mesh_tets(mesh);
faces = pde_mesh_faces(mesh);

fprintf('Cylinder cantilever, 200 km/h wind:\n');
fprintf('  R = %.3f m, H = %.3f m, voxel = %.3f m\n', R, H, voxel);
fprintf('  q_dyn = %.0f Pa, p_wind = %.0f Pa\n', q_dyn, p_wind);
fprintf('  mesh:  %.0f nodes, %.0f tets, %.0f boundary faces\n', ...
        size(nodes, 1), size(tets, 1), size(faces, 1));

% --- Linear elasticity assembly -----------------------------------
E  = 6.9e10;
nu = 0.33;
K = pde_assemble_elast_3d_sparse(mesh, E, nu);

% --- Wind load + clamp --------------------------------------------
% Cylinder axis is +z (from 0 to H); wind blows in +y so it hits
% face 3 (-y side of the cylinder).  Clamp face 1 (the z=0 base).
F = pde_face_pressure_3d(mesh, 3.0, p_wind);
fixed_nodes = pde_face_nodes(mesh, 1.0);
sys2 = pde_apply_fixed_3d_sparse(K, F, fixed_nodes);

Kc = pde_sys_K_sparse(sys2);
Fc = pde_sys_F(sys2);

% --- Sparse PCG solve ----------------------------------------------
res    = pcg(Kc, Fc, 1.0e-5, 20000.0);
u      = pcg_x(res);
flag   = pcg_flag(res);
iters  = pcg_iter(res);
relres = pcg_relres(res);

% --- Post-process: per-node vM + render ---------------------------
vm_node = pde_node_von_mises_3d(mesh, u, E, nu);
disp    = pde_reshape_disp_3d(u);
def     = pde_peak_disp_3d(u);

% Peak vM via simple loop.
nn = size(vm_node, 1);
peak_vm = 0.0;
for i = 1:nn
    v = vm_node(i);
    if v > peak_vm
        peak_vm = v;
    end
end

% --- Analytic comparison ------------------------------------------
A_proj    = 2.0 * R;                       % proj width (per unit length)
w_load    = p_wind * A_proj;               % N/m distributed load
M_max     = w_load * H * H / 2.0;          % N·m at clamped base
I_section = 3.14159265358979 * R*R*R*R / 4.0;
sigma_an  = M_max * R / I_section;         % Pa
delta_an  = w_load * H*H*H*H / (8.0 * E * I_section);

fprintf('\nFEM solve:\n');
fprintf('  PCG flag:           %.0f (0=converged)\n', flag);
fprintf('  PCG iterations:     %.0f\n', iters);
fprintf('  PCG relres:         %.2e\n', relres);
fprintf('  peak FEM disp:      %.3f mm\n', def * 1000);
fprintf('  peak FEM vM:        %.3f MPa\n', peak_vm / 1e6);

fprintf('\nAnalytic cantilever (Euler-Bernoulli):\n');
fprintf('  w load:             %.1f N/m\n', w_load);
fprintf('  M_max @ base:       %.1f N*m\n', M_max);
fprintf('  I (pi R^4 / 4):     %.3e m^4\n', I_section);
fprintf('  sigma_max:          %.3f MPa\n', sigma_an / 1e6);
fprintf('  delta_tip:          %.3f mm\n', delta_an * 1000);

fprintf('\nFEM / analytic ratios:\n');
fprintf('  displacement ratio: %.2f\n', (def) / delta_an);
fprintf('  stress ratio:       %.2f\n', (peak_vm) / sigma_an);

% --- Render with same percentile-clamp colour map -----------------
peak_marker = peak_vm;
clamp_max = peak_marker * 0.05;            % less aggressive for cyl
ext_vm = zeros(nn, 1);
for i = 1:nn
    v = vm_node(i);
    if v > clamp_max
        v = clamp_max;
    end
    ext_vm(i) = v;
end

pdeplot3d_deform_scale(200.0);
pdeplot3d_deformation(disp);
pdeplot3d(nodes, faces, ext_vm);
title('Cylinder cantilever 200 km/h wind, vM clamped at 5% of peak');
saveas(gcf, '/tmp/cylinder_wind.png');
fprintf('\n  rendered:           /tmp/cylinder_wind.png\n');
