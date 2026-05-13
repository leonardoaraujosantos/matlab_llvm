% pde_thermal_stress.m — thermal-stress coupling via cellLoad.
%
% Two-step workflow:
%   1) Solve thermal steady-state on a bar with one face at 100 C
%      and the opposite face at 0 C (linear gradient).
%   2) Solve structural static on the same mesh, clamping x=0 and
%      letting the bar expand under the thermal field.  Expected
%      free-end expansion ~ α · ΔT_avg · L.
%
% For steel with α = 12e-6 / K, ΔT_avg = 50 C across L = 0.1 m:
%   u = α ΔT_avg L = 12e-6 × 50 × 0.1 = 6e-5 m = 0.06 mm
% (axial displacement at the free end, plus geometric & Poisson
% redistribution).

gm = pde_mesh_cuboid_tet(0.1, 0.02, 0.02, 5, 1, 1);

% --- thermal solve ----------------------------------------------------
m_t = femodel('AnalysisType', 'thermalSteadyState', 'Geometry', gm);
m_t = pde_set_material(m_t, ...
            materialProperties('ThermalConductivity', 50.0));
m_t = pde_set_face_temperature(m_t, 5, 100.0);
m_t = pde_set_face_temperature(m_t, 6,   0.0);
m_t = pde_generate_mesh(m_t);
R_t = pde_solve(m_t);

% --- structural with thermal-stress coupling --------------------------
m_s = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
m_s = pde_set_material(m_s, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850, ...
                               'CTE',           12e-6));
m_s = pde_set_face_fixed(m_s, 5);            % clamp x=0
m_s = pde_set_reference_temperature(m_s, 0.0);
m_s = pde_set_cell_temperature(m_s, R_t);
m_s = pde_generate_mesh(m_s);
R_s = pde_solve(m_s);
u   = pde_kernel_u(R_s);

n = size(u, 1) / 3;
peak_ux = 0.0;
for i = 1:n
    ux = u(3*i - 2);
    if ux > peak_ux
        peak_ux = ux;
    end
end

fprintf('PDE thermal-stress coupling (cellLoad Temperature=R_t):\n');
fprintf('  peak free-end u_x (mm):   %.3f\n', peak_ux * 1000);
fprintf('  ~analytic alpha dT L (mm): 0.060\n');
