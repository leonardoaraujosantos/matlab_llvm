% Regression fixture for #124.
%
% Under `matlabc -dap` (whole-file ReplMode) a struct / class-instance
% variable mutated *in place* by BARE method calls across statements must
% stay visible to a later read.  This mirrors examples/pde/poisson_disk.m
% (Poisson on the unit disk, analytic u(0)=0.25) trimmed of the plot:
%
%   model = createpde();              % stored to the workspace, not a slot
%   geometryFromEdges(model, g);      % bare call: mutates model in place
%   ... model.Geometry.NumEdges ...   % field access -> creates a read-cache
%                                     %   struct slot, blank-inited at entry
%   specifyCoefficients(model, ...);  % bare call: mutates model in place
%   generateMesh(model, Hmax=0.05);   % bare call: mutates model in place
%   R = solvepde(model);              % must see the meshed model
%
% Before the fix, every read of `model` AFTER the field access loaded the
% stale blank slot instead of the live workspace pointer, so `solvepde`
% ran on an empty model and printed U0=0.0000.  The fix routes struct/obj
% read-cache bindings to the workspace at every read site.
g = decsg([1; 0; 0; 1; 0; 0; 0; 0; 0; 0]);

model = createpde();
geometryFromEdges(model, g);

applyBoundaryCondition(model, "dirichlet", Edge=1:model.Geometry.NumEdges, ...
                       u=0);
specifyCoefficients(model, m=0, d=0, c=1, a=0, f=1);
generateMesh(model, Hmax=0.05);
R = solvepde(model);

u_center = interpolateSolution(R, 0, 0);
fprintf('U0=%.4f\n', u_center);
