# Partial Differential Equation Toolbox — Tutorial

The PDE Toolbox runtime compiles MATLAB finite-element workflows down through MLIR to native code. It covers the classic 2-D elliptic loop (`createpde` → `decsg` → `solvepde`), the unified 3-D `femodel` structural workflow, and a function-form FEM stack for large voxelized/STL geometries solved with sparse iterative solvers. The headline is a 3-D wind-stress study that reports von Mises stress and tip deflection on a deformed mesh.

## Supported features

- **2-D elliptic FEM**: `createpde`, `decsg`, `geometryFromEdges`, `applyBoundaryCondition` (Dirichlet), `specifyCoefficients` (`m,d,c,a,f`), `generateMesh(Hmax=…)`, `solvepde`, `interpolateSolution`, `pdeplot` (with `Contour`).
- **3-D `femodel` classdef workflow**: `multicuboid`, `femodel(AnalysisType="structuralStatic"|"structuralModal", Geometry=…)`, `materialProperties`, `faceBC(Constraint="fixed")`, `faceLoad(Pressure=…)`, `generateMesh`, `solve`, result fields `R.Displacement.{Magnitude,ux,uy,uz}`, `R.VonMisesStress`, `R.NaturalFrequencies`, `R.ModeShapes.Magnitude`, `pdeplot3D(ColorMapData=…, Deformation=…, DeformationScaleFactor=…)`.
- **Function-form FEM core** (large meshes): `pde_mesh_cuboid_tet`, `pde_multicylinder`, geometry import `pde_load_glb` / `pde_load_stl` / `pde_save_stl`, voxelizer `pde_voxelize_surface`, mesh accessors `pde_mesh_nodes` / `pde_mesh_tets` / `pde_mesh_faces`.
- **3-D elasticity assembly + loads + BCs**: `pde_assemble_elast_3d`, `pde_assemble_elast_3d_sparse`, `pde_face_pressure_3d`, `pde_face_nodes`, `pde_apply_fixed_3d` / `pde_apply_fixed_3d_sparse`, system accessors `pde_sys_K` / `pde_sys_K_sparse` / `pde_sys_F`.
- **Sparse solvers**: dense `Kc \ Fc`, and sparse `pcg(K, F, tol, maxit)` with `pcg_x` / `pcg_flag` / `pcg_iter` / `pcg_relres`; sparse metadata `sprows`, `spnnz`.
- **Post-processing + render**: `pde_von_mises_3d`, `pde_node_von_mises_3d`, `pde_reshape_disp_3d`, `pde_peak_disp_3d`, `pdeplot3d` with `pdeplot3d_deform_scale` / `pdeplot3d_deformation`, `saveas`.

## Build & run

```bash
build/matlabc -emit-llvm examples/pde/wind_stress_3d.m > /tmp/wind_stress_3d.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/wind_stress_3d.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/wind_stress_3d
/tmp/wind_stress_3d
```

## Worked examples

### The headline: 3-D wind-stress study  (`examples/pde/wind_stress_3d.m`)

A 3 m × 0.05 m × 2 m steel sign-panel is clamped on its left face and hit by a 250 km/h aerodynamic wind pressure on its front face. The script builds a tet mesh, assembles 3-D linear elasticity, applies the load and clamp, solves, then recovers von Mises stress and peak displacement.

```matlab
% Wind: q_dyn = 0.5*rho*v^2, p_wind = Cd*q_dyn
p_wind = Cd * q_dyn;

W = 3.0; D = 0.05; H = 2.0;
mesh = pde_mesh_cuboid_tet(W, D, H, 12, 1, 8);

E = 2.0e11; nu = 0.30;
K = pde_assemble_elast_3d(mesh, E, nu);

F = pde_face_pressure_3d(mesh, 3.0, p_wind);   % face 3 = y=0 front
fixed_nodes = pde_face_nodes(mesh, 5.0);       % face 5 = x=0 left
sys2 = pde_apply_fixed_3d(K, F, fixed_nodes);

Kc = pde_sys_K(sys2);
Fc = pde_sys_F(sys2);
u  = Kc \ Fc;

vm  = pde_von_mises_3d(mesh, u, E, nu);
def = pde_peak_disp_3d(u);
```

The mesh has 234 nodes / 576 tets / 702 DOFs, small enough for a dense `Kc \ Fc`. The script then scans `vm` for its peak (there is no scalar-returning vector-max on this lane yet) and prints peak displacement in mm, peak von Mises in MPa, and the safety factor against S275 steel's 275 MPa yield.

### Cantilever cylinder vs. closed form  (`examples/pde/cylinder_wind_stress.m`)

A solid aluminium cylinder (R = 0.05 m, H = 1.91 m) clamped at the base under 200 km/h wind. This is the analytic sanity check: the FEM result is compared against the Euler-Bernoulli cantilever formulas for bending moment, section stress, and tip deflection.

```matlab
mesh = pde_multicylinder(R, H, voxel);
K = pde_assemble_elast_3d_sparse(mesh, E, nu);
F = pde_face_pressure_3d(mesh, 3.0, p_wind);
sys2 = pde_apply_fixed_3d_sparse(K, F, pde_face_nodes(mesh, 1.0));

res    = pcg(pde_sys_K_sparse(sys2), pde_sys_F(sys2), 1.0e-5, 20000.0);
u      = pcg_x(res);
peak_vm = ...                          % loop-scan over pde_node_von_mises_3d

M_max     = w_load * H * H / 2.0;
I_section = pi * R^4 / 4.0;
sigma_an  = M_max * R / I_section;     % analytic bending stress
```

Because the cylinder mesh is large, it uses the sparse assembly + `pcg` path. The script prints the FEM/analytic ratios for both displacement and stress — they land close to 1, validating the sparse elasticity pipeline.

### Poisson on the unit disk  (`examples/pde/poisson_disk.m`)

The smallest 2-D elliptic gating loop, solving `-Δu = 1` with `u = 0` on the boundary against the exact solution `u(r) = (1 − r²)/4`.

```matlab
g = decsg([1; 0; 0; 1; 0; 0; 0; 0; 0; 0]);   % unit circle, DG format
model = createpde();
geometryFromEdges(model, g);
applyBoundaryCondition(model, "dirichlet", Edge=1:model.Geometry.NumEdges, u=0);
specifyCoefficients(model, m=0, d=0, c=1, a=0, f=1);
generateMesh(model, Hmax=0.05);
R = solvepde(model);
u_center = interpolateSolution(R, 0, 0);      % ~0.25
```

This is the full `createpde` → `decsg` → coefficients → mesh → `solvepde` → `interpolateSolution` → `pdeplot` chain. The reported error at the centre vs. 0.25 confirms convergence.

### Clamped plate via `femodel`  (`examples/pde/clamped_plate_pressure.m`)

The MATLAB-faithful classdef workflow: a thin plate built with `multicuboid`, clamped on its side faces, with 1 MPa on the top.

```matlab
gm    = multicuboid(1.0, 1.0, 0.01);
model = femodel(AnalysisType="structuralStatic", Geometry=gm);
model.MaterialProperties = materialProperties(YoungsModulus=2.0e11, ...
                              PoissonsRatio=0.30, MassDensity=7850);
model.FaceBC(1:4) = faceBC(Constraint="fixed");
model.FaceLoad(6) = faceLoad(Pressure=1.0e6);
model = generateMesh(model, Hmax=0.02);
R     = solve(model);

peak_def = max(R.Displacement.Magnitude);
peak_vm  = max(R.VonMisesStress);
```

Note the `femodel` lane exposes `R.Displacement.{Magnitude,ux,uy,uz}` and `R.VonMisesStress` directly, and `pdeplot3D` accepts a `Deformation` struct of `ux/uy/uz`.

### Tuning-fork modal analysis  (`examples/pde/tuningfork_modal.m`)

`structuralModal` analysis on an STL-imported solid solves the generalised eigenproblem `K φ = ω² M φ`, returning natural frequencies and mode shapes.

```matlab
model = femodel(AnalysisType="structuralModal", Geometry="fixtures/TuningFork.stl");
model.MaterialProperties = materialProperties(YoungsModulus=210e9, ...
                              PoissonsRatio=0.30, MassDensity=8000);
model = generateMesh(model, Hmax=0.001);
RF = solve(model, FrequencyRange=[-Inf, 4000] * 2 * pi);
fHz = RF.NaturalFrequencies / (2 * pi);
pdeplot3D(RF.Mesh, ColorMapData=RF.ModeShapes.Magnitude(:, 7));
```

The first six near-zero modes are rigid-body; mode 7 is the first flexible mode.

### Remaining examples

- `antenna_glb_fem.m` and `antenna_wind_stress.m` — the full real-world pipeline: `pde_load_glb` → `pde_voxelize_surface` → sparse elasticity → clamp + face pressure → `pcg` → per-node vM → `pdeplot3d` with deformation exaggeration (the wind demo even bakes a wind-direction arrow into the rendered mesh and computes the safety factor against Al 6061-T6 yield).
- `stl_load_render.m` — STL round-trip: `pde_save_stl` → `pde_load_stl` → render coloured by node height, validating geometry import/export.

## Limitations & carve-outs

From [`docs/pde_toolbox_roadmap.md`](../pde_toolbox_roadmap.md):

- Tiers 1–4 (function-form numeric core: 2-D elliptic, 3-D elasticity, transient/thermal/electrostatic/magnetostatic, ROM, nonlinear) are shipped. Dense linear algebra is the default; the sparse + `pcg` path handles the large voxelized/STL meshes.
- The MATLAB-faithful `femodel`/`solvepde` classdef façade is the Tier-5 polish layer; some Tier-2/3 examples use the lower-level `pde_*` function-form API instead.
- Mode shapes for some eigen-analyses were a deferred follow-up (Lanczos arc); `structuralModal` with mode-shape recovery is in the production-solver arc.
- Out of scope: the Econometric/PDE Modeler **app** GUI, Live-Editor tasks, and the `batteryP2DModel` (Tier-5 specialty) lane.

## See also

- Roadmap: [`docs/pde_toolbox_roadmap.md`](../pde_toolbox_roadmap.md)
- Examples: `examples/pde/` (see also `examples/pde/README.md` for the per-tier gating table)
