% pde_quad_tet.m — quadratic-tet (T10) elasticity vs P1 linear tet.
%
% Two sanity checks on a 0.5m × 50mm × 50mm steel bar:
%   (a) Uniaxial pull along the long axis — both T4 and T10 should
%       recover the analytic u = σL/E.
%   (b) Cantilever bending — T10 should yield a LARGER tip
%       displacement than T4 on the same mesh (P1 over-stiffness
%       in bending).

gm = pde_mesh_cuboid_tet(0.5, 0.05, 0.05, 4, 1, 1);

E  = 2.0e11;
nu = 0.30;
p  = 1.0e6;  % 1 MPa pull

% -- T10 path via direct kernels.
mq    = pde_mesh_quadratic(gm);
K_sp  = pde_assemble_elast_3d_t10(mq, E, nu);
% Face 6 is +x (far end); pull negative pressure = outward traction.
F     = pde_face_pressure_3d_t10(mq, 6, -p);
fixed = pde_face_nodes_t10(mq, 5);   % clamp x=0 face
sys2  = pde_apply_fixed_3d_sparse(K_sp, F, fixed);
Kc    = pde_sys_K(sys2);
Fc    = pde_sys_F(sys2);
gr    = sparse_gmres_ilu0(Kc, Fc, 1e-10, 4000);
u10   = pcg_x(gr);

% Peak |u_x| over all nodes — the +x face elongation.
n10 = size(u10, 1) / 3;
peak_x = 0.0;
for i = 1:n10
    ux = u10(3*i - 2);
    if ux > peak_x
        peak_x = ux;
    end
end

% Analytic: u = σ L / E.  For p = 1 MPa, L = 0.5 m, E = 2e11 Pa:
%   u = 1e6 * 0.5 / 2e11 = 2.5e-6 m = 0.0025 mm.
u_anal_mm = p * 0.5 / E * 1000;

% -- Reference T4 path on the same mesh.
m4 = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
m4 = pde_set_material(m4, ...
        materialProperties('YoungsModulus', E, ...
                           'PoissonsRatio', nu, ...
                           'MassDensity',   7850));
m4 = pde_set_face_fixed   (m4, 5);
m4 = pde_set_face_pressure(m4, 6, -p);
m4 = pde_generate_mesh(m4);
R4 = pde_solve(m4);
u4 = pde_kernel_u(R4);
n4 = size(u4, 1) / 3;
peak4 = 0.0;
for i = 1:n4
    ux = u4(3*i - 2);
    if ux > peak4
        peak4 = ux;
    end
end

fprintf('PDE Quadratic-tet (T10), uniaxial pull:\n');
fprintf('  T10 mesh nodes:           %.0f\n', n10);
fprintf('  T10 max u_x (mm):         %.4f\n', peak_x * 1000);
fprintf('  T4  max u_x (mm):         %.4f\n', peak4 * 1000);
fprintf('  analytic u (mm):          %.4f\n', u_anal_mm);

% (b) Cantilever bending: top-face uniform pressure, x=0 clamped.
pb = 1.0e5;

% T10 path:
K10b = pde_assemble_elast_3d_t10(mq, E, nu);
F10b = pde_face_pressure_3d_t10(mq, 2, pb);
fix10 = pde_face_nodes_t10(mq, 5);
sys_b = pde_apply_fixed_3d_sparse(K10b, F10b, fix10);
gr_b  = sparse_gmres_ilu0(pde_sys_K(sys_b), pde_sys_F(sys_b), 1e-10, 4000);
u10b  = pcg_x(gr_b);
peak_10b = 0.0;
for i = 1:n10
    uz = u10b(3*i);
    if uz < 0
        uz = -uz;
    end
    if uz > peak_10b
        peak_10b = uz;
    end
end

% T4 path:
m4b = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
m4b = pde_set_material(m4b, ...
        materialProperties('YoungsModulus', E, ...
                           'PoissonsRatio', nu, ...
                           'MassDensity',   7850));
m4b = pde_set_face_fixed   (m4b, 5);
m4b = pde_set_face_pressure(m4b, 2, pb);
m4b = pde_generate_mesh(m4b);
R4b = pde_solve(m4b);
u4b = pde_kernel_u(R4b);
peak_4b = 0.0;
for i = 1:n4
    uz = u4b(3*i);
    if uz < 0
        uz = -uz;
    end
    if uz > peak_4b
        peak_4b = uz;
    end
end

% Report rounded ratios — exact values depend on Kuhn-mesh diagonals
% and aren't expected to be 1:1, but T10/T4 should exceed 1 (less
% stiff in bending).
flag = 0;
if peak_10b > peak_4b
    flag = 1;
end
fprintf('  bending T4 peak |uz|(mm):  %.4f\n', peak_4b * 1000);
fprintf('  bending T10 peak |uz|(mm): %.4f\n', peak_10b * 1000);
fprintf('  T10 > T4 (less stiff):    %.0f\n', flag);
