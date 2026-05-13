% clamped_plate_pressure.m — clamped 3-D plate under a uniform
% pressure.  Smoke test for the 3-D linear-elasticity assembly +
% pressure-load surface integral.  See docs/pde_toolbox_roadmap.md
% §3.4 (Tier-2 row).
%
% Status: 🔵 not started.
%
% Geometry: 1 m × 1 m × 0.01 m thin steel plate, clamped on all four
% side faces, with a 1 MPa pressure on the top face.

gm = multicuboid(1.0, 1.0, 0.01);

model = femodel(AnalysisType="structuralStatic", Geometry=gm);
model.MaterialProperties = ...
    materialProperties(YoungsModulus=2.0e11, ...
                       PoissonsRatio=0.30, ...
                       MassDensity=7850);

% Faces 1..4 are the four side faces of the plate.
model.FaceBC(1:4) = faceBC(Constraint="fixed");

% Face 6 is the top face.  Apply 1 MPa pressure (positive = into the
% body, i.e. pushing down on the top).
model.FaceLoad(6) = faceLoad(Pressure=1.0e6);

model = generateMesh(model, Hmax=0.02);
R     = solve(model);

% Centre-of-plate deflection (peak displacement).
peak_def = max(R.Displacement.Magnitude);
peak_vm  = max(R.VonMisesStress);
fprintf('Peak deflection:        %.3f mm\n', peak_def * 1000);
fprintf('Peak von Mises stress:  %.2f MPa\n', peak_vm / 1e6);

% Plot stress on the deformed plate.
defs = struct('ux', R.Displacement.ux, ...
              'uy', R.Displacement.uy, ...
              'uz', R.Displacement.uz);
pdeplot3D(R.Mesh, ColorMapData=R.VonMisesStress, ...
          Deformation=defs, DeformationScaleFactor=500);
title('Clamped plate under uniform 1 MPa pressure');
