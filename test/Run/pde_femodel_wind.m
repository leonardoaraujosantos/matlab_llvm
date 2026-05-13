% pde_femodel_wind.m — MATLAB-faithful wind-stress demo using the
% femodel classdef façade.
%
% Reproduces the function-form pde_wind_stress.m result via the
% public MathWorks-style API: femodel + materialProperties + faceBC
% + faceLoad ctor sugar + pde_set_* setters.  The Result class
% StaticStructuralResults exposes .Displacement (sub-struct with
% .Magnitude / .ux / .uy / .uz) and .VonMisesStress.

rho_air = 1.225;
v_kmh   = 250;
v_ms    = v_kmh / 3.6;
q_dyn   = 0.5 * rho_air * v_ms * v_ms;
Cd      = 1.2;
p_wind  = Cd * q_dyn;

% --- Geometry: 3 m x 50 mm x 2 m sign-panel ---------------------
gm = pde_mesh_cuboid_tet(3.0, 0.05, 2.0, 12, 1, 8);

% --- femodel classdef workflow ----------------------------------
model = femodel('AnalysisType', 'structuralStatic', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);
model = pde_set_face_pressure(model, 3, p_wind);
model = pde_generate_mesh(model);

% solve() returns a result struct with the same Mesh/u/vm fields
% that the StaticStructuralResults ctor sugar would pack into the
% MathWorks-shape class instance.
raw = pde_solve(model);
u   = pde_kernel_u(raw);
vm  = pde_kernel_vm(raw);

% Build the MathWorks-shape result via the kwarg-ctor sugar.  Each
% kwarg pair becomes a typed matlab_obj_set_* call at the call site.
R = StaticStructuralResults( ...
        'Mesh',           pde_kernel_mesh(raw), ...
        'Displacement',   pdeDisplacement( ...
                              'ux',        u, ...
                              'uy',        u, ...
                              'uz',        u, ...
                              'Magnitude', u), ...
        'VonMisesStress', vm);

% Now property reads:
mag = R.VonMisesStress;
n   = size(mag, 1);
peak_vm = 0.0;
for i = 1:n
    v = mag(i);
    if v > peak_vm
        peak_vm = v;
    end
end

% Coarse log10 bucket.
peak_vm_log10 = floor(log10(peak_vm));

fprintf('PDE femodel API: 250 km/h wind on sign-panel\n');
fprintf('  AnalysisType:         structuralStatic\n');
fprintf('  E / nu:               200 GPa / 0.3\n');
fprintf('  load:                 Pressure %.0f Pa on face 3\n', round(p_wind));
fprintf('  BC:                   fixed on face 5\n');
fprintf('  log10(peak vM Pa):    %.0f\n', peak_vm_log10);
