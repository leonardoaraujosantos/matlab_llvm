% wind_stress_3d.m — THE HEADLINE Tier-2 demo (function-form).
%
% A 3 m x 0.05 m x 2 m steel sign-panel is fixed on its left face
% (x = 0) and loaded with a 250 km/h aerodynamic wind pressure on
% its front face (y = 0).  Reports peak von Mises stress and peak
% displacement, validating where the structure is most stressed.
%
% Wind physics:
%   rho_air = 1.225 kg/m^3        (sea-level standard atmosphere)
%   v       = 250 km/h = 69.444 m/s
%   q_dyn   = 0.5 * rho * v^2 = 2952.45 Pa  (free-stream dynamic pressure)
%   Cd      = 1.2                 (flat-plate normal drag)
%   p_wind  = Cd * q_dyn = 3543 Pa
%
% Uses the Tier-2 function-form API.  The MATLAB-faithful
% `femodel(AnalysisType="structuralStatic", ...)` classdef wrapper is
% deferred Tier-5 polish — see docs/pde_toolbox_roadmap.md.
%
% Build + run:
%   ./matlabc -emit-llvm examples/pde/wind_stress_3d.m > /tmp/wind.ll
%   /opt/homebrew/opt/llvm/bin/clang++ /tmp/wind.ll \
%       runtime/matlab_runtime.cpp runtime/runtime_debug.cpp \
%       runtime/runtime_complex.cpp runtime/runtime_comm.cpp \
%       runtime/runtime_prop.cpp runtime/runtime_rf.cpp \
%       runtime/runtime_pde.cpp -I runtime -o /tmp/wind && /tmp/wind

% --- Wind load ------------------------------------------------------
rho_air = 1.225;
v_kmh   = 250;
v_ms    = v_kmh / 3.6;
q_dyn   = 0.5 * rho_air * v_ms * v_ms;
Cd      = 1.2;
p_wind  = Cd * q_dyn;

fprintf('Wind:   %.0f km/h, dynamic pressure %.0f Pa, effective %.0f Pa\n', ...
        v_kmh, q_dyn, p_wind);

% --- Geometry: 3 m wide x 0.05 m thick x 2 m tall panel ------------
W = 3.0; D = 0.05; H = 2.0;
% Mesh: 12 x 1 x 8 hex cells -> 234 nodes, 576 tets, 702 DOFs.
mesh = pde_mesh_cuboid_tet(W, D, H, 12, 1, 8);

% --- Structural steel ---------------------------------------------
E  = 2.0e11;     % Young's modulus, Pa
nu = 0.30;       % Poisson's ratio

K = pde_assemble_elast_3d(mesh, E, nu);

% Wind pressure on face 3 (y = 0, the front of the panel).
F = pde_face_pressure_3d(mesh, 3.0, p_wind);

% Clamp the left edge (face 5, x = 0).  In a real install this would
% be where the panel attaches to a vertical post.
fixed_nodes = pde_face_nodes(mesh, 5.0);
sys2 = pde_apply_fixed_3d(K, F, fixed_nodes);

Kc = pde_sys_K(sys2);
Fc = pde_sys_F(sys2);
u  = Kc \ Fc;

vm  = pde_von_mises_3d(mesh, u, E, nu);
def = pde_peak_disp_3d(u);

% Scan vM for the peak (no vector-max builtin path yet that returns f64).
n_tets = size(vm, 1);
peak_vm = 0.0;
for i = 1:n_tets
    val = vm(i);
    if val > peak_vm
        peak_vm = val;
    end
end

fprintf('\nPeak displacement:  %.3f mm\n', def * 1000);
fprintf('Peak von Mises:     %.2f MPa\n',  peak_vm / 1e6);
fprintf('Yield (S275 steel): 275 MPa\n');
fprintf('Safety factor:      %.1f\n', 275e6 / peak_vm);
