% blade_pitch_opt.m — Optimization Toolbox Tier-2 headline demo.
%
% Cross-toolbox showcase: a 3-D linear-elasticity FEM solve (PDE
% Toolbox) characterises a wind-turbine blade-root segment, and
% `fmincon` (Optimization Toolbox) then chooses the blade pitch that
% maximises aerodynamic power without exceeding the structural stress
% limit.
%
% Problem
% -------
%   maximise   aerodynamic power  P(theta)
%   subject to peak von Mises stress  sigma_vM(theta) <= sigma_limit
%              pitch bounds           0 <= theta <= 25  degrees
%
% Coupling
% --------
% Step 1 runs the full PDE Tier-2 pipeline
%   mesh -> assemble_elast_3d -> face_pressure -> apply_fixed
%        -> mldivide -> von_mises
% on the blade-root segment under a reference windward pressure, and
% extracts the stress-per-unit-pressure coefficient k_stress.  Because
% linear elasticity makes peak stress exactly proportional to the
% applied load, sigma_vM(theta) = k_stress * p_wind(theta) is an
% *exact* surrogate — so Step 2's `fmincon` run uses the FEM-derived
% coefficient directly (anonymous-function constraints cannot capture
% workspace variables, so the value is written as a literal and the
% script asserts it matches the live FEM result).
%
% See docs/optim_toolbox_roadmap.md and docs/pde_toolbox_roadmap.md.

% ---- Step 1: FEM characterisation of the blade-root segment ------
p_ref = 9.0e3;                                   % reference pressure, Pa
mesh  = pde_mesh_cuboid_tet(2.0, 0.04, 0.30, 6, 2, 2);
K     = pde_assemble_elast_3d(mesh, 7.0e10, 0.33);   % aluminium
F     = pde_face_pressure_3d(mesh, 6.0, p_ref);      % windward face
fixed = pde_face_nodes(mesh, 5.0);                   % clamped root
sys   = pde_apply_fixed_3d(K, F, fixed);
u     = pde_sys_K(sys) \ pde_sys_F(sys);
vm    = pde_von_mises_3d(mesh, u, 7.0e10, 0.33);
s_ref = max(vm);
k_stress = s_ref / p_ref;                        % stress per unit load

% k_stress from the FEM solve (~1.00947); used as a literal below.
disp(round(k_stress * 1000));                    % -> 1009

% ---- Step 2: fmincon — maximise power within the stress limit ----
% Windward pressure model:  p_wind(theta) = 9000 * (1 + 0.0016*theta^2)
% Stress surrogate:         sigma(theta)  = k_stress * p_wind(theta)
% Stress limit 13500 Pa binds at a pitch well below the 25 deg bound,
% while the unconstrained power optimum sits near 35 deg.  Pitch is in
% degrees; the power surrogate converts to radians (deg2rad = pi/180).
obj = @(p) -(1.0e5 * sin(p(1) * 0.017453292519943) ...
             * cos(p(1) * 0.017453292519943) ...
             * cos(p(1) * 0.017453292519943));
con = @(p) [1.00947 * 9000 * (1 + 0.0016 * p(1) * p(1)) - 13500];
theta = fmincon(obj, 5, [], [], [], [], 0, 25, con);

topt  = theta(1);
s_opt = 1.00947 * 9000 * (1 + 0.0016 * topt * topt);
disp(round(topt));                               % optimal pitch, deg

% ---- self-checks (robust to FEM floating-point jitter) -----------
% 1. The literal k_stress matches the live FEM coefficient.
if abs(k_stress - 1.00947) < 1e-2; disp(1); else; disp(0); end

% 2. The optimal pitch respects the lower bound (the upper bound is
%    covered by check 5; each test is a single comparison because the
%    LLVM lane does not lower `&&` as a value).
if topt >= -1e-6; disp(1); else; disp(0); end

% 3. The stress constraint is satisfied at the optimum.
if s_opt <= 13500 + 1.0; disp(1); else; disp(0); end

% 4. The stress constraint is the binding one (active, not the bound):
%    the optimiser pushed the pitch up until stress hit the limit.
if s_opt >= 13500 - 50; disp(1); else; disp(0); end

% 5. The optimum is strictly below the 25 deg pitch bound.
if topt < 24; disp(1); else; disp(0); end
