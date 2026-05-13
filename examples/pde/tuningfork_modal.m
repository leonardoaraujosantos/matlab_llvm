% tuningfork_modal.m — natural frequencies + mode shapes of a tuning
% fork (STL-imported solid geometry).
%
% This example mirrors the MATLAB docs "Structural Dynamics of Tuning
% Fork" sample.  It gates the Tier-2 STL importer + the
% `structuralModal` analysis type + the generalised eigenvalue solver
% `K φ = ω² M φ` (see docs/pde_toolbox_roadmap.md §3.3, §10.5).
%
% Status: 🔵 not started.
%
% Required runtime asset: examples/pde/fixtures/TuningFork.stl
% (binary STL of a U-shaped tuning fork; not yet shipped — placeholder
% lives in the fixtures directory once Tier-2 lands).

% Material: structural steel.
E   = 210e9;
nu  = 0.30;
rho = 8000;

model = femodel(AnalysisType="structuralModal", ...
                Geometry="fixtures/TuningFork.stl");
model.MaterialProperties = ...
    materialProperties(YoungsModulus=E, PoissonsRatio=nu, MassDensity=rho);

model = generateMesh(model, Hmax=0.001);

% Solve for all modes with circular frequencies up to 4000 Hz · 2π.
% The −Inf lower bound captures the six rigid-body modes near zero.
RF = solve(model, FrequencyRange=[-Inf, 4000] * 2 * pi);

% Express natural frequencies in Hz and list them.
fHz = RF.NaturalFrequencies / (2 * pi);
fprintf(' Mode    Frequency (Hz)\n');
for k = 1:numel(fHz)
    fprintf('  %2d        %10.3f\n', k, fHz(k));
end

% First flexible mode is mode 7 (rigid-body 1..6 are near zero).
% Plot its magnitude.
pdeplot3D(RF.Mesh, ColorMapData=RF.ModeShapes.Magnitude(:, 7));
title(sprintf('Mode 7: %.1f Hz', fHz(7)));
