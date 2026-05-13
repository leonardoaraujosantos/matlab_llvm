% poisson_disk.m — Poisson's equation on the unit disk.
% Smallest 2-D elliptic FEM gating example.  Validates Tier-1 of
% docs/pde_toolbox_roadmap.md: geometry → mesh → assemble → solve.
%
%   -Δu = 1   in the unit disk,
%      u = 0   on the boundary.
%
% Analytic solution: u(r) = (1 - r^2) / 4.
%
% Status: 🔵 not started.

% Decomposed-geometry matrix for a single circle of radius 1 centred
% at the origin.  DG format: column = [shape_id; x; y; r; ...].
g = decsg([1; 0; 0; 1; 0; 0; 0; 0; 0; 0]);

model = createpde();
geometryFromEdges(model, g);

% Default Dirichlet u = 0 on all edges.
applyBoundaryCondition(model, "dirichlet", Edge=1:model.Geometry.NumEdges, ...
                        u=0);

% PDE: -∇·(1·∇u) + 0·u = 1
specifyCoefficients(model, m=0, d=0, c=1, a=0, f=1);

generateMesh(model, Hmax=0.05);
R = solvepde(model);

% Compare at the centre against the analytic value u(0) = 0.25.
u_center = interpolateSolution(R, 0, 0);
fprintf('FEM   u(0) = %.6f\n', u_center);
fprintf('Exact u(0) = 0.250000\n');
fprintf('Error      = %.2e\n', abs(u_center - 0.25));

% Plot
pdeplot(model, XYData=R.NodalSolution, Contour="on");
title('Poisson u = (1 - r^2) / 4 on the unit disk');
