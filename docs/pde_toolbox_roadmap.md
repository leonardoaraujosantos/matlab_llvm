# Partial Differential Equation Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Partial Differential Equation Toolbox
programs.

Source: *Partial Differential Equation Toolbox User's Guide* (R2026a,
2308 pages, 5 chapters: Getting Started · Setting Up Your PDE ·
Solving PDEs · PDE Modeler App · Functions).

The headline target end-user demo (the gating example for this whole
roadmap) is in [`examples/pde/wind_stress_3d.m`](../examples/pde/wind_stress_3d.m):
*import or construct a 3-D model, apply a 250 km/h aerodynamic wind
pressure on the windward face, and visualize the von Mises stress map
to see which parts of the structure are most loaded.* Achieving that
demo end-to-end is what closes **PDE-Tier-2** below.

Companion docs: [`feature_status.md`](feature_status.md),
[`roadmap.md`](roadmap.md), [`ode.md`](ode.md) (where the existing
`pdepe` 1-D solver lives), [`plotting.md`](plotting.md) (where the
3-D mesh / deformation rendering work plugs in).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order.  Tier-0
  is what we already have (`pdepe` 1-D MOL).  Tier-1 / 2 close the
  smallest useful 2-D + 3-D linear-static loops.  Tier-3+ adds modal,
  transient, thermal, EM, nonlinear, ROM, PINN.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
- **REPL / Debug** rows note display + DAP variable-inspector
  expectations.  The PDE toolbox introduces several new descriptor
  types (`PDEModel`, `femodel`, `FEMesh`, `StationaryResults`,
  `StaticStructuralResults`, `ModalStructuralResults`, …) — each
  needs a renderer in `runtime_debug.cpp` and an opt-into-the-DAP
  child-walker for the result fields.
- **Compile/Execute**: every Tier-N row crosses Sema (Resolver
  registers new builtins) → MLIR (`matlab.call_builtin` rewrites
  to runtime entries via a new `LowerPDE.cpp` pass) → Runtime
  (`runtime/runtime_pde.cpp`) → JIT (resolved via the shared
  `DynamicLibrarySearchGenerator`, same as `runtime_signal`, etc.).
- **Debug/REPL**: the symbolic-style classdef path used by RF
  (`RFSparameters` / `RFCktAmplifier`) is the model. `femodel` is
  effectively a typed bag of fields whose accessors are sugar over
  function-form runtime entries — exactly the precedent we already
  have.

---

## 1. Already shipped (Tier-0 → Tier-4 function-form, 2026-05-13)

### Tier-0 — 1-D method-of-lines (pre-existing)

These pieces are wired through Sema → MLIR → LLVM/C/C++/Python/TS
lanes today.

| Group | Surface | Location | Notes |
|---|---|---|---|
| 1-D PDE method-of-lines | `pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)` | `runtime/matlab_runtime.cpp` §pdepe | Cartesian (m=0) / cylindrical (m=1) / spherical (m=2); Dirichlet, Neumann, Robin BCs; scalar PDE; non-uniform mesh.  Wraps `ode23s_v` for stiff time integration.  Gating: `test/Run/math_pdepe_heat.m`, `math_pdepe_neumann.m`, `math_pdepe_radial.m`. |
| Stiff time integration | `ode23s` (Rosenbrock 2(3)), `ode15s` (planned) | `runtime/matlab_runtime.cpp` §ode | The natural backbone for any *transient* PDE problem after spatial discretisation.  Already vector-y, FSAL, RelTol/AbsTol/MaxStep/Jacobian-FD. |
| Linear algebra primitives | `mldivide`, `expm`, `eig` (full+symmetric), `schur`, `hess`, `lu_decompose`, `chol` (planned) | `runtime/matlab_runtime.cpp` | Dense-only today.  Sufficient for FEM problems with up to ~5 k DOF (the dense K factorisation is ~O(n³) but still seconds at that scale).  Sparse is on the PDE-Tier-1 critical path — see §10.1 below. |
| Classdef machinery | property kwarg-ctor sugar, typed getters, `RFCktAmplifier`-style hierarchies | `runtime/matlab_runtime.cpp` §classdef + `runtime/cst_classdefs.m` style | We will use exactly this pattern for `femodel`, `materialProperties`, `faceBC`, `faceLoad`, etc. |
| Plotting | `surf`, `surf3`, `plot3`, `mesh`, `contour`, `quiver`, `quiver3`, colormap LUTs | `runtime/plot/` | Cairo headless backend; PNG/SVG/PDF.  Does **not** yet handle the unstructured-mesh case (`trisurf` / `tetramesh` / `pdeplot` / `pdeplot3D`) — see §10.2. |
| Symbolic `pdsolve` | closed-form heat / wave / 1st-order linear via SymPP | `lib/Sym/` | Useful for verification of the FEM approximation against analytic solutions. |

**Coverage today closes a one-spatial-dimension subset.**  Everything
in the PDE Toolbox User's Guide that is 1-D parabolic-elliptic
(§3 Heat Distribution in Circular Cylindrical Rod, §3 Inhomogeneous
Heat Equation on Square Domain projected to 1-D, §3 Nonlinear Heat
Transfer in Thin Plate restricted to 1-D radial, the basic Wave
Equation slice) can already be solved by `pdepe`.

### Tier-1 → Tier-4 function-form (shipped 2026-05-13)

Per the 4-tier tracer-bullet plan agreed in chat.  Function-form
numeric core lives in `runtime/runtime_pde.cpp`; classdef wrappers
(`createpde`, `femodel`, `materialProperties`, `faceLoad`, …) are
deferred Tier-5 polish.  All gating tests pass under
`test/Run/run_tests.sh` (LLVM lane only — emit-c / cpp / python / ts
lanes carry `.skip-emit-*` markers).

#### Shipped runtime entries

**Tier-1 (2-D scalar elliptic)** — `runtime/runtime_pde.cpp`:
- `matlab_pde_mesh_rect_tri(x0, x1, y0, y1, Nx, Ny)` — structured
  triangulation of `[x0,x1] × [y0,y1]`.
- `matlab_pde_boundary_nodes_rect(mesh)` — 1-based node ids on the
  rectangle boundary.
- `matlab_pde_assemble_poisson_2d(mesh, c, a, f)` — P1 linear triangle
  assembly returning struct `{K, F}`.
- `matlab_pde_apply_dirichlet(sys, node_ids, u_val)` — row-zero +
  diag-1 enforcement of fixed-value DOFs.
- `matlab_pde_sys_K` / `matlab_pde_sys_F` — field accessors.
- `matlab_pde_mesh_nodes` / `matlab_pde_mesh_triangles` — mesh
  accessors.

**Tier-2 (3-D linear elasticity)** — `runtime/runtime_pde.cpp`:
- `matlab_pde_mesh_cuboid_tet(W, D, H, Nx, Ny, Nz)` — structured
  hex grid split 6-tet/hex on a `[0,W] × [0,D] × [0,H]` cuboid.
  Face id convention: 1=z=0, 2=z=H, 3=y=0, 4=y=D, 5=x=0, 6=x=W.
- `matlab_pde_face_nodes(mesh, face_id)` — unique node ids on a face.
- `matlab_pde_assemble_elast_3d(mesh, E, nu)` — 4-node tet
  constant-strain element K (3N × 3N) for isotropic linear
  elasticity.
- `matlab_pde_face_pressure_3d(mesh, face_id, pressure)` — surface
  pressure load distributed via piecewise-linear basis (positive p
  acts into the body along the inward normal).
- `matlab_pde_apply_fixed_3d(K, F, node_ids)` — clamp all 3 DOFs of
  the specified nodes.
- `matlab_pde_von_mises_3d(mesh, u, E, nu)` — per-element von Mises
  recovered from σ = D · B · u_e.
- `matlab_pde_peak_disp_3d(u)` — peak displacement magnitude (max
  over nodes of |[ux,uy,uz]|).
- `matlab_pde_mesh_tets` / `matlab_pde_mesh_faces` — mesh accessors.

**Tier-3 (2-D transient + modal)** — `runtime/runtime_pde.cpp`:
- `matlab_pde_assemble_transient_2d(mesh, c, a, f)` — assembles M, K,
  F together; M is lumped-diagonal for forward-Euler stability.
- `matlab_pde_init_uniform_2d(mesh, u_init, bnd)` — IC vector
  (uniform interior + zero on the boundary).
- `matlab_pde_step_forward_euler_2d(M, K, F, u, bnd, dt)` — one
  explicit time step with post-step Dirichlet re-enforcement.
- `matlab_pde_eigsmall(K, M, nmodes)` — k smallest generalised
  eigenvalues of `K φ = λ M φ` via inverse iteration with M-orthogonal
  deflation (adequate for ≤ 300 DOF).
- `matlab_pde_sys_M` — extra accessor.

**Tier-4 (nonlinear)** — `runtime/runtime_pde.cpp`:
- `matlab_pde_solve_nonlinear_2d(mesh, c0, alpha, f, c_func)` —
  Picard iteration outer loop on `-∇·(c(u)∇u) = f`.  `c_func=1`
  selects `c(u) = c0·(1 + α u²)`.
- `matlab_pde_result_solution` / `_num_iters` / `_resid` — accessors.

#### Sema + MLIR wiring

- `lib/Sema/Resolver.cpp` registers 32 new builtins under a single
  per-toolbox block (`pde_mesh_rect_tri`, …, `pde_result_resid`).
- `lib/Sema/TypeInference.cpp` types matrix-returning entries as
  `Array(Double)`, scalar entries as `scalar(Double)`.
- `lib/MLIR/Passes/LowerTensorOps.cpp` adds a 32-entry dispatch table
  inside `rewriteBuiltinCalls()` that rewrites each `matlab.call_builtin`
  to the corresponding `matlab_pde_*` `llvm.call`.  The dispatcher
  *loosely* matches operand types — `tensor<*xf64>` operands at sites
  that expect `PtrTy` are bridged with `mlir::UnrealizedConversionCastOp`,
  unblocking cross-block uses inside `matlab.for` loop bodies that
  the intra-block slot promoter can't lift.
- Field accessors (`pde_sys_K`, `pde_sys_F`, `pde_sys_M`,
  `pde_mesh_nodes`, etc.) sidestep the Sema default of "struct field
  read → f64" that would otherwise zero out matrix-valued fields.

#### Gating tests (all green on the LLVM lane)

| Test | Tier | Validates |
|---|:-:|---|
| `test/Run/pde_poisson_square.m` | 1 | 2-D Poisson on unit square: u(0.5,0.5) = 0.0735 vs analytic 0.0737 |
| `test/Run/pde_clamped_plate.m` | 2 | 3-D cantilever beam under top-face pressure: 135 µm tip deflection |
| `test/Run/pde_wind_stress.m` | 2 | **Headline:** 3 m × 50 mm × 2 m sign-panel under 250 km/h wind; reports log10(peak vM) = 6 Pa (~MPa range, well below 275 MPa yield) |
| `test/Run/pde_heat_transient.m` | 3 | u_t = Lap(u) with u(x,y,0)=1: centre decays from 1.0000 → 0.0312 at t=0.2 (analytic ≈ 0.0314) |
| `test/Run/pde_nonlinear.m` | 4 | Picard iteration on c(u)=1+5u²: converges in 4 iters, u(0.5,0.5) = 0.0729 vs linear 0.0737 |
| `test/Runtime/test_pde.c` | 1+2 | Direct C runtime test of the FEM core, bypassing the frontend / JIT |

CTest target: `runtime-tests-pde` (built via `cmake --build . --target runtime-test-pde`).

#### Geometry import + 3-D rendering (shipped 2026-05-13, follow-up arc)

After the Tier-1 → Tier-4 numeric core landed, a second arc shipped
the two highest-leverage practical-visualisation items: STL/GLB
import + unstructured-mesh 3-D rendering.

**STL importer** (`matlab_pde_load_stl`, `matlab_pde_save_stl`):
- ASCII + binary auto-detect via file-size sniff + "solid" keyword.
- Vertex welding via hash-by-quantised-coordinate (`llrint(x · 1e9)`).
- Round-trip safe: save a mesh, load it back, get the same surface.
- Tier-2 cuboid round-trip recovers 26 unique surface vertices from
  a 2 × 2 × 2 cell mesh + 48 boundary triangles.

**GLB importer** (`matlab_pde_load_glb`):
- glTF 2.0 binary container: 12-byte header + JSON chunk + BIN chunk.
- In-tree minimal JSON parser (~200 LOC) walks
  `meshes[0].primitives[0].attributes.POSITION` + `indices`.
- Supports `componentType` 5121 (uint8) / 5123 (uint16) / 5125
  (uint32) / 5126 (float).  Single-mesh, mode=4 (TRIANGLES) only —
  TRIANGLE_STRIP / FAN are out of v1 scope.
- No scene-graph TRS transforms — adequate for single-asset
  visualisation.

**pdeplot3D** (`matlab_pdeplot3d`, `matlab_pdeplot3d_deformation`,
`matlab_pdeplot3d_deform_scale`):
- Cairo unstructured-mesh painter with Painter's-algorithm depth
  sort, Lambertian shading (40 % ambient + 60 % directional), and
  colormap-by-vertex (Gouraud-averaged per triangle) or
  colormap-by-face.
- Accepts both the volumetric-mesh `Faces` table from `multicuboid`
  (Nf × 4: `[face_id n1 n2 n3]`) and the pure-surface mesh from
  STL/GLB (Nf × 4 with face_id = 1).
- Deformation: per-vertex displacement vector applied
  pre-projection, scale-factor controlled by a sticky per-thread
  setter so calls like `pdeplot3d_deform_scale(100); pdeplot3d(...)`
  Just Work.

**Test infrastructure**:
- `test/Runtime/test_pde_io.c` — direct-C STL/GLB round-trip tests
  (4 cases: ASCII triangle, binary tetrahedron, cuboid round-trip
  with surface vertex count check, GLB quad). Wired as
  `runtime-tests-pde_io` in ctest.
- `test/Run/pde_wind_plot.m` — end-to-end Sema → MLIR → Cairo render
  of the wind-stress vM map.  Marked `.requires-plot` so the harness
  conditionally links the Cairo runtime.
- `examples/pde/stl_load_render.m` — STL save → load → render demo.

**Test harness extension**: `test/Run/run_tests.sh` now reads
`<name>.requires-plot` markers and conditionally links
`runtime/plot/*.cpp` + Cairo via pkg-config when present, SKIPping
tests on machines without Cairo.

#### Architectural simplifications taken (deliberate)

These are spelled out so future contributors don't re-pay the design cost:
- **Dense matrices throughout.** No sparse-matrix subsystem.  K is
  stored as a full `N × N` dense `matlab_mat *`; the dense `mldivide`
  LU is the linear solver.  Practical ceiling is ~3 000 DOF (~3000² ·
  8 B ≈ 72 MB for K alone).  Sparse + Krylov is roadmap §10.1 and
  unblocks problems at ~50 k DOF.
- **Structured tet meshes** for `multicuboid` (no Delaunay /
  Bowyer-Watson).  Each hex cell is split into 6 tetrahedra via
  Kuhn's "0-6 diagonal" decomposition.  Trades mesh quality for
  shipping speed.  `multicylinder` / `multisphere` / `fegeometry`
  from arbitrary STL volumetric mesher are deferred to roadmap §3.1.
- **STL/GLB are surface-only.**  The importers produce a
  triangulated surface mesh suitable for visualisation but not for
  volumetric FEM solve.  Wrapping the surface to a tet mesh
  (TetGen-style) is a separate roadmap §10.3 sub-project.
- **No `femodel` classdef.**  The MATLAB-faithful kwarg API
  (`femodel(AnalysisType="structuralStatic", Geometry=…)`) is deferred
  to roadmap §3.3.  Today users write function-form code:
  `K = pde_assemble_elast_3d(mesh, E, nu)` etc.  The numeric core is
  identical to what `femodel` would dispatch to under the hood.
- **No sparse Krylov modal solver.**  `pde_eigsmall` uses dense
  inverse iteration with M-orthogonal deflation.  Fine for ≤ 300 DOF
  modal problems; Krylov-Schur / Lanczos with shift-invert is
  roadmap §10.5.
- **pdeplot3D shading is per-triangle.**  Gouraud shading on
  unstructured meshes needs Cairo's mesh-gradient pattern; for v1
  each triangle is filled with the mean of its three vertex values
  through the colormap, with Lambertian shading from a fixed light
  direction.  Adequate for stress/temperature/voltage maps at the
  mesh densities the current dense linalg can handle.

Nothing in the toolbox's *2-D / 3-D geometry → FEM mesh → assemble →
solve* pipeline used to be wired before this arc.  Now Tiers 1-4 are
in place as function-form numerics with end-to-end gating tests.
The remaining roadmap is the classdef façade, sparse infra, the
unstructured-mesh plotting, and the production-quality mesher / STL
importer.

---

## 2. Tier 1 — 2-D elliptic, the smallest end-to-end FEM loop (~3 wk)

This tier closes the smallest *useful* loop a new PDE user expects to
work: write down `–∇·(c∇u) + au = f` on a 2-D domain, generate a
triangular mesh, apply Dirichlet/Neumann BCs, solve, and plot.  It is
also a pure-prerequisite tier — none of Tier-2 / Tier-3+ can ship
without the mesher, the assembler, and the sparse linear solve that
live here.

The user-facing surface is the **legacy / general PDE workflow**:
`createpde` → `geometryFromEdges` → `applyBoundaryCondition` →
`specifyCoefficients(m=, d=, c=, a=, f=)` → `generateMesh` →
`solvepde` → `pdeplot`.

### 2.1 Geometry — 2-D Decomposed Geometry (DG) format

| Feature | Status | Notes |
|---|:-:|---|
| Built-in basic shapes — `R1` rectangle, `C1` circle, `E1` ellipse, `P1` polygon, `SQ1` square | 🔵 | Same DG matrix format MATLAB uses: each column encodes the shape type + parameters.  The matrix is a plain `double[10, N]`. |
| `decsg(gd, sf, ns)` — Decomposed Solid Geometry from a *Set formula* like `"R1+C1-C2"` | 🔵 | Pure numeric — boolean-of-half-planes per element.  Hardest part is segmenting the resulting outline into edges with consistent normals. |
| `geometryFromEdges(model, g)` — bind DG → `PDEModel.Geometry` | 🔵 | |
| `pdegplot(g, EdgeLabels="on")` | 🔵 | Wraps the existing 2-D `plot` runtime with label overlays.  Builtin: `pdegplot(g, kwargs...)`. |
| `polyshape` → DG conversion | 🔵 | Useful for the pillow-block-bearing-style examples that build a 2-D outline as a `polyshape` and rotate-extrude it into 3-D. |

### 2.2 Mesh — 2-D triangulation

| Feature | Status | Notes |
|---|:-:|---|
| `generateMesh(model, Hmax=, Hmin=, Hgrad=)` returning a `FEMesh` | 🔵 | Constrained Delaunay triangulation with quality refinement.  We can either (a) bring in a tiny in-tree mesher like Triangle's "ruppert" algorithm (~600 LOC C, BSD-licensable but watch the license tail), or (b) write a frontal-Delaunay (Persson's `distmesh` style — ~150 LOC, fully in-house, slower but simple).  Option (b) is the chosen path: it lives in `runtime/runtime_pde_mesh.cpp` and depends only on `mldivide` + bbox/segment intersection. |
| `refineMesh(model)`, `adaptmesh(model)` | 🔵 | Edge bisection + a-posteriori error estimator; needed for Tier-3 adaptive Poisson example. |
| `findNodes(mesh, "region", "Edge", k)` / `findElements(mesh, "region", "Face", k)` | 🔵 | Required by every example that applies a non-trivial load. |
| `pdemesh(mesh)`, `pdemesh(p, e, t)` | 🔵 | Wireframe triangle render — wraps Cairo `line_segment` from `runtime/plot/`. |
| `meshData` triple `[p, e, t]` | 🔵 | `p` = nodes (2 × Np), `e` = edges (7 × Ne), `t` = triangles (4 × Nt).  Internal format inherited from MATLAB. |

### 2.3 Coefficients, BCs, ICs

| Feature | Status | Notes |
|---|:-:|---|
| `specifyCoefficients(model, m, d, c, a, f)` | 🔵 | Scalar PDE form: `m·∂²u/∂t² + d·∂u/∂t − ∇·(c∇u) + au = f`.  Coefficients may be scalar, 2×2 (anisotropic c), or functions of `(location, state)`.  Constant + function forms in this tier; full N-component systems land in Tier-3. |
| `applyBoundaryCondition(model, "dirichlet"\|"neumann"\|"mixed", Edge=…, u=…, g=…, q=…, h=…, r=…)` | 🔵 | Both constant and `@(location, state)` forms. |
| `setInitialConditions(model, u0, ut0)` | 🔵 | Needed for the 2-D Wave-Equation example. |

### 2.4 Sparse-matrix infrastructure (critical prerequisite)

| Feature | Status | Notes |
|---|:-:|---|
| `matlab_mat_sparse` opaque type (new descriptor kind) | 🔵 | CSR layout (`int64_t *row_ptr`, `int64_t *col_idx`, `double *data`).  Living in `runtime/runtime_internal.h` next to the existing `matlab_mat` / `matlab_mat_c` family. |
| `sparse(rows, cols, vals, m, n)` constructor | 🔵 | Triplet-form constructor, `m × n` shape declared explicitly. |
| Sparse `+ - * \` arithmetic | 🔵 | Sparse × sparse via row-merged accumulation; sparse `\` dense via UMFPACK-style LU (in-tree minimal symbolic-numeric column pivoting — Davis's textbook reference, ~400 LOC).  Iterative fallback: BiCGSTAB + ILU(0) for problems > 50 k DOF. |
| `pcg`, `bicgstab`, `gmres`, `minres` Krylov | 🔵 | Builtins on `(A, b)` returning `x` (+ optional `flag`, `relres`, `iter`). |
| `assembleFEMatrices(model, "KMA"\|"nullspace")` | 🔵 | The actual FEM assembly entry point — returns a struct `(K, M, F, Q, G, H, R)` ready to feed `K\F`.  This is what unblocks every Tier-2 example. |

### 2.5 Linear solvers — first cut

| Feature | Status | Notes |
|---|:-:|---|
| `solvepde(model)` stationary | 🔵 | Calls `assembleFEMatrices` then either dense `mldivide` (small N) or sparse LU (large N).  Returns a `StationaryResults` classdef with `NodalSolution`, `XGradients`, `YGradients`, `Mesh`. |
| `solvepde(model, tlist)` parabolic / hyperbolic | 🔵 | Calls `assembleFEMatrices`, then `ode23s_v` (parabolic) or a Newmark-β integrator (hyperbolic).  Returns `TimeDependentResults`. |
| `solvepdeeig(model, range)` | 🔵 | Generalised symmetric eigenvalue `K v = λ M v` via subspace iteration or shift-invert ARPACK-style.  Returns `EigenResults` with `Eigenvalues`, `Eigenvectors`, `Mesh`. |

### 2.6 Plotting — 2-D solution rendering

| Feature | Status | Notes |
|---|:-:|---|
| `pdeplot(model, XYData=u, Contour="on"\|"off", FlowData=[ux uy])` | 🔵 | Wraps `trisurf` on the 2-D triangulation: colour per node, Gouraud-shaded across the triangle. |
| `pdegplot(model, EdgeLabels=, FaceLabels=)` | 🔵 | Just the geometry + labels. |
| `pdemesh(p, e, t)` | 🔵 | Wireframe triangle plot. |

### 2.7 Tier-1 result classes (new descriptor types)

| Class | Properties | DAP renderer |
|---|---|---|
| `PDEModel` | `Geometry`, `Mesh`, `EquationCoefficients`, `BoundaryConditions`, `InitialConditions`, `IsTimeDependent` | Expandable struct-of-handles render. |
| `FEMesh` | `Nodes`, `Elements`, `MaxElementSize`, `MinElementSize`, `MeshGradation`, `GeometricOrder` | Render summary: "FEMesh (12 348 nodes, 23 117 triangles, Hmax=0.012)". |
| `StationaryResults` | `NodalSolution`, `XGradients`, `YGradients`, `ZGradients`, `Mesh` | Same. |
| `TimeDependentResults` | `NodalSolution(t)`, `SolutionTimes`, `Mesh` | Render as "TimeDependentResults: N_t=…, N_nodes=…". |
| `EigenResults` | `Eigenvalues`, `Eigenvectors`, `Mesh` | Same. |

### 2.8 Gating examples for Tier 1

`examples/pde/poisson_disk.m` — `–Δu = 1` on the unit disk with `u = 0`
on the boundary (compare to the analytic `(1−r²)/4`).

`examples/pde/poisson_lshape.m` — Poisson's equation on the L-shaped
membrane.  Validates the corner-singularity treatment of `refineMesh`.

`examples/pde/eig_lshape_membrane.m` — eigenvalues + eigenmodes of the
L-shaped membrane (the classic MATLAB logo problem); validates
`solvepdeeig`.

`examples/pde/wave_square.m` — Wave equation `u_tt = Δu` on the unit
square with Dirichlet BC and a one-bump initial displacement.
Validates the 2-D hyperbolic transient path.

---

## 3. Tier 2 — 3-D unified structural workflow + the headline wind-stress demo (~3 wk)

This is where the **user's stated goal lives**: a 3-D model, an
applied 250 km/h wind, and a von-Mises-stress visualisation.  Tier 2
takes the Tier-1 numeric core and adds (a) 3-D geometry + tet mesher,
(b) the modern **unified `femodel` workflow** (vs. the legacy
`createpde` workflow which is also already covered by Tier-1's
generic `specifyCoefficients`), (c) linear elasticity assembly, and
(d) the 3-D rendering primitives.

### 3.1 3-D geometry

| Feature | Status | Notes |
|---|:-:|---|
| `multicuboid(W, D, H)`, `multicuboid([W1 W2], D, H)` stacked-cube | 🔵 | Returns a `fegeometry` classdef wrapping a triangulated surface mesh.  Builtin: `multicuboid(...)`. |
| `multicylinder(R, H, Void=…)` | 🔵 | Single + nested-radius variant; `Void=[true,false]` is hollow inner.  Used by the pillow-block-bearing example. |
| `multisphere(R)` | 🔵 | Solid sphere. |
| `multicuboid` + boolean union via STL stitching | 🔵 | Required for the "pillow block" / "cat" composite examples — but **not** on the Tier-2 critical path; can be deferred to Tier-4. |
| `fegeometry(nodes, elements)` from triangulated mesh | 🔵 | The fallback path for *any* 3-D shape we cannot synthesise from primitives — including the user's headline demo if they bring an STL file. |
| `importGeometry(model, "file.stl")` — **STL ASCII + binary import** | 🔵 | Critical for letting the user drop in their own 3-D model.  Pure-C parser (`runtime/runtime_pde_stl.cpp`) — STL is trivially simple (header + N×(normal + 3 vertices) records).  Output goes through a vertex-welding pass (hash-by-quantized-coordinate) into the same `fegeometry` triangulated-surface representation. |
| `importGeometry(model, "file.step")` STEP import | 🔵 | Deferred to Tier-5 (STEP is BREP not mesh — needs `OpenCASCADE` or a STEP→tessellation library; non-trivial license tail). |
| `translate`, `rotate`, `scale` on `fegeometry` | 🔵 | Affine transforms on the triangulated surface. |
| `addVertex(gm, Coordinates=[x y z])`, `addFace`, `addCell` | 🔵 | Lets users place vertices for `vertexLoad` / point-force application. |

### 3.2 3-D mesh — tetrahedra

| Feature | Status | Notes |
|---|:-:|---|
| `generateMesh(model, Hmax=, Hmin=, Hgrad=, GeometricOrder="linear"\|"quadratic")` returning a 3-D `FEMesh` | 🔵 | Tet mesher.  Two routes: (a) **TetGen** (BSD-with-attribution, ~14 k LOC C++, well-tested), or (b) in-tree **Delaunay-of-Bowyer-Watson + Lawson edge-flips** with octree-driven refinement (~800 LOC, slower, no licence tail).  Default to (b) for the licence-clean path; expose `MATLAB_LLVM_WITH_TETGEN=ON` opt-in for production-quality meshes on large geometries. |
| Quadratic tetrahedra (10-node) | 🔵 | The user-facing default in MATLAB; needed for accurate stress recovery. |
| `findElements(mesh, "region", "Cell", k)` | 🔵 | Required for `cellLoad`. |

### 3.3 Unified `femodel` workflow (the modern PDE Toolbox API)

The modern API is a single classdef whose `AnalysisType` field
switches between problem families:

```matlab
model = femodel(AnalysisType="structuralStatic", Geometry=gm);
model.MaterialProperties = materialProperties(YoungsModulus=210e9, ...
                                              PoissonsRatio=0.3, ...
                                              MassDensity=7850);
model.FaceBC(1)    = faceBC(Constraint="fixed");
model.FaceLoad(2)  = faceLoad(Pressure=2952);   % 250 km/h wind
model              = generateMesh(model, Hmax=0.05);
R                  = solve(model);
pdeplot3D(R.Mesh, ColorMapData=R.VonMisesStress);
```

| Classdef | Properties | Constructed via |
|---|---|---|
| `femodel` | `AnalysisType`, `Geometry`, `Mesh`, `MaterialProperties`, `FaceBC[]`, `EdgeBC[]`, `VertexBC[]`, `FaceLoad[]`, `EdgeLoad[]`, `VertexLoad[]`, `CellLoad[]`, `FaceIC[]`, `PlanarType` | kwarg-ctor sugar — `femodel(AnalysisType=..., Geometry=...)` |
| `materialProperties` | `YoungsModulus`, `PoissonsRatio`, `MassDensity`, `ThermalConductivity`, `SpecificHeat`, `RelativePermittivity`, `RelativePermeability`, `ElectricalConductivity` | kwarg-ctor |
| `faceBC` / `edgeBC` / `vertexBC` | `Constraint` ∈ {"fixed", "roller", "free"}, `Displacement`, `XDisplacement`, `YDisplacement`, `ZDisplacement`, `Temperature`, `Voltage` | kwarg-ctor |
| `faceLoad` / `edgeLoad` / `vertexLoad` / `cellLoad` | `Pressure`, `SurfaceTraction`, `Force`, `Heat`, `Temperature`, `CurrentDensity`, `ChargeDensity` | kwarg-ctor |
| `faceIC` / `cellIC` | `Displacement`, `Velocity`, `Temperature` | kwarg-ctor |

**AnalysisType strings to support in Tier 2** (others land in Tier 3+):
`"structuralStatic"`, `"structuralModal"` (added end of Tier 2 for the
modal sanity-check).

### 3.4 Linear elasticity FEM assembly

| Feature | Status | Notes |
|---|:-:|---|
| 3-D Cauchy / Lamé constitutive `D = (E/((1+ν)(1−2ν))) · [...]` for isotropic isothermal | 🔵 | 6 × 6 stress-strain matrix; standard textbook form. |
| 4-node (linear) and 10-node (quadratic) tetrahedral element K-matrix assembly | 🔵 | Gauss quadrature (1-point for linear K, 4-point for quadratic). |
| Plane stress / plane strain 2-D path | 🔵 | `model.PlanarType = "planeStress"` toggles the 3×3 D-matrix.  Lands as Tier-2 sub-row because the 2-D coverage from Tier-1 generic-PDE doesn't include the vector-valued elasticity tensor `c ⊗ ∇u`. |
| Pressure-load assembly on a face — `f = ∫_face p · n dA` | 🔵 | Pressures default along the **inward face normal**; matches MATLAB.  Critical for the wind-stress demo. |
| Surface-traction load — `faceLoad(SurfaceTraction=[Tx;Ty;Tz])` | 🔵 | Vector form; user-specified traction vector. |
| Vertex/edge point loads | 🔵 | Lumped onto the nearest node. |
| Body / cell load (gravity, centrifugal) | 🔵 | `cellLoad(Force=[0;0;-rho*g])`. |

### 3.5 Result objects + post-processing

| Class | Properties | Notes |
|---|---|---|
| `StaticStructuralResults` | `Displacement.{ux,uy,uz,Magnitude}`, `Stress.{xx,yy,zz,xy,yz,xz}`, `Strain.{xx,yy,zz,xy,yz,xz}`, `VonMisesStress`, `Mesh` | The 6 stress components are stored per node (recovered from element-Gauss-point values via L²-projection). |
| `interpolateDisplacement(R, query)` | Trilinear-in-tet barycentric interpolation. | |
| `interpolateStress(R, query)` | Same; uses the projected nodal stresses. | |
| `evaluateStrain(R, query)`, `evaluateVonMisesStress(R, query)`, `evaluatePrincipalStress(R, query)` | All same path. | |

### 3.6 Plotting — 3-D solution rendering

| Feature | Status | Notes |
|---|:-:|---|
| `pdeplot3D(mesh, ColorMapData=…, Deformation=…, DeformationScaleFactor=…)` | 🔵 | Renders the boundary triangulation of the tet mesh, colour by `ColorMapData` (e.g. `R.VonMisesStress`), optionally displaced by `Deformation`.  Backend: extend `runtime/plot/cairo_render.cpp` with a `trisurf3d` painter — Painter's-algorithm depth sort over the boundary triangles, Lambertian shading + colormap per face.  No OpenGL.  See §10.2 for the full plot-runtime delta. |
| `pdeviz(mesh, nodalData, ...)` (Live Editor task) | 🔵 | Same renderer; just a different kwarg surface.  Animation loop in `examples/pde/wind_stress_3d.m` is done by writing N PNGs and assembling with `ffmpeg` — we will not ship an interactive viewer in Tier 2. |
| `pdegplot(gm, FaceLabels="on")` for 3-D | 🔵 | Surface render of `fegeometry` + numbered labels per face/edge. |

### 3.7 Gating examples for Tier 2

`examples/pde/clamped_plate_pressure.m` — clamped square 3-D isotropic
plate with uniform pressure load.  Validates `faceBC(Constraint="fixed")`
+ `faceLoad(Pressure=…)` and von Mises stress recovery against the
known analytical centre-deflection.

`examples/pde/bracket_deflection.m` — L-bracket fixed on one face,
traction on the opposite face; validates `SurfaceTraction` vector
form.

**`examples/pde/wind_stress_3d.m` — the headline demo.**  A 3-D
geometry built from `multicuboid` (or imported from STL) representing
a tall flat sign-panel + post (or any model the user supplies),
clamped at the base, loaded with the dynamic pressure from a 250 km/h
wind on the windward face:

```matlab
% Air-density and wind speed
rho_air = 1.225;                 % kg / m^3
v_kmh   = 250;
v_ms    = v_kmh / 3.6;           % 69.444 m / s
q_dyn   = 0.5 * rho_air * v_ms^2;  % ≈ 2952 Pa dynamic pressure
Cd      = 1.2;                   % drag coeff for a flat plate normal
p_wind  = Cd * q_dyn;            % effective pressure on the windward face

% Geometry: 3 m × 0.05 m × 2 m sign-panel on a 0.1 m square × 4 m post
panel = multicuboid(3.0, 0.05, 2.0, 'ZOffset', 4.0);
post  = multicuboid(0.1, 0.1, 4.0);
gm    = stitchGeometry(panel, post);   % helper, see Tier-2 §3.1

model = femodel(AnalysisType="structuralStatic", Geometry=gm);
model.MaterialProperties = ...
    materialProperties(YoungsModulus=2.0e11, PoissonsRatio=0.30, MassDensity=7850);
model.FaceBC(1)   = faceBC(Constraint="fixed");          % base of the post
model.FaceLoad(2) = faceLoad(Pressure=p_wind);           % windward face of the panel
model             = generateMesh(model, Hmax=0.05);
R                 = solve(model);

% Von-Mises stress map on the deformed shape
defs = struct('ux', R.Displacement.ux, ...
              'uy', R.Displacement.uy, ...
              'uz', R.Displacement.uz);
pdeplot3D(R.Mesh, ColorMapData=R.VonMisesStress, ...
          Deformation=defs, DeformationScaleFactor=200);

% Numerical summary
fprintf('Peak von Mises stress: %.2f MPa\n', max(R.VonMisesStress)/1e6);
fprintf('Peak displacement:     %.3f mm\n',  max(R.Displacement.Magnitude)*1000);
```

This is **the** demo for the toolbox.  Achieving it end-to-end —
compile, JIT, execute, REPL inspect `R`, debug-stop and DAP-render
`R.VonMisesStress`, and have the PNG land — is the Tier-2 done-bar.

`examples/pde/wing_spar_static.m` — I-beam wing spar (`multicuboid`
+ `addFace` to carve the I-cross-section), fixed on one end,
distributed face traction; validates the structural-static path on a
more complex (multi-face) load surface.

`examples/pde/tuningfork_modal.m` — STL-imported tuning-fork geometry,
`structuralModal` analysis, table of natural frequencies, plot of
mode-7 shape.  Validates `solvepdeeig` on the 3-D elasticity operator.

---

## 4. Tier 3 — Transient + frequency-response structural, thermal, electromagnetics (~3 wk)

### 4.1 Structural — transient + modal + frequency-response

| Feature | Status | Notes |
|---|:-:|---|
| `AnalysisType="structuralTransient"` | 🔵 | `M Ü + C U̇ + K U = F(t)`.  Newmark-β integrator (β=0.25, γ=0.5 default) on the sparse system.  Falls back to `ode23s_v` if `M` is singular. |
| `AnalysisType="structuralModal"` (Tier-2 sub-row promoted here) | 🔵 | `K φ = ω² M φ` generalised eig via subspace iteration; ARPACK-style if large.  Already needed for the tuning-fork example. |
| `AnalysisType="structuralFrequency"` | 🔵 | Harmonic response — `(K − ω² M + iωC) U = F`, complex sparse solve per ω. |
| `solve(model, ModalResults=RF)` modal superposition | 🔵 | Project to modal subspace, integrate in time, reconstruct nodal solution. |
| Rayleigh damping `[α β]` + critical-damping % | 🔵 | `C = α M + β K`. |
| `cellLoad(Temperature=…)` thermal stress | 🔵 | Couples thermal field as a body load. |

### 4.2 Thermal — steady-state + transient

| AnalysisType | What |
|---|---|
| `"thermalSteadyState"` | `−∇·(k ∇T) = Q`; `faceLoad(Heat=…)`, `faceBC(Temperature=…)`. |
| `"thermalTransient"` | `ρ c_p ∂T/∂t − ∇·(k ∇T) = Q`; couples to `ode23s_v`. |
| Nonconstant `k(T)` | Picard iteration outer loop; uses the existing iterative-Newton infrastructure. |
| Surface-to-surface radiation | View-factor matrix assembly (geometric); couples as a quadratic boundary term. |

### 4.3 Electromagnetics — electrostatic + magnetostatic

| AnalysisType | What |
|---|---|
| `"electrostatic"` | `−∇·(ε ∇V) = ρ`; `faceBC(Voltage=…)`, `cellLoad(ChargeDensity=…)`. |
| `"magnetostatic"` | 2-D Poisson on `A_z` (scalar magnetic vector potential); 3-D curl-curl on `A` (needs Nédélec edge elements — defer to Tier-5). |
| `"dcConduction"` | `−∇·(σ ∇V) = 0`; `edgeLoad(SurfaceCurrentDensity=…)`. |
| `"harmonicElectromagnetic"` (scattering) | Helmholtz on `E_z` (TE) / `H_z` (TM). |

### 4.4 Gating examples for Tier 3

`examples/pde/heat_sink_transient.m` — heat-sink finite element model
with state-space simulation (couples to the existing CST `ss` class).

`examples/pde/jet_turbine_thermal_stress.m` — thermal then structural
analysis chained, with `cellLoad(Temperature=Rt)` feeding from a
thermal-transient solution.

`examples/pde/electrostatic_busbar.m` — electrostatic analysis of a
transformer bushing insulator.

`examples/pde/magnetostatic_2pole_motor.m` — 2-D magnetostatic on a
two-pole electric motor.

`examples/pde/tuningfork_transient.m` — transient-response analysis
chained off the Tier-2 modal results.

---

## 5. Tier 4 — Reduced-Order Models + nonlinear + adaptive (~2 wk)

| Feature | Status | Notes |
|---|:-:|---|
| `reduce(model, FrequencyRange=…)` Craig-Bampton ROM | 🔵 | Wing-spar example.  Output is a `ReducedStructuralModel` with `K`, `M`, `R` matrices; couples to Simulink Descriptor-State-Space (out of scope) but the model object itself is in scope. |
| `reconstructSolution(reducedR, x_t)` | 🔵 | |
| Nonlinear stationary — `c(u, ∇u)` Picard / Newton outer loop | 🔵 | Already prototyped for `pdepe` nonlinear path. |
| `adaptmesh` — adaptive mesh refinement | 🔵 | Edge-bisection + element error estimator. |
| Geometric nonlinear elasticity (large deformation) | 🔵 | `cCoefficientLagrangePlaneStress` helper from the clamped-beam example. |
| Cross-coupled multi-physics nonlinear systems | 🔵 | `createpde(N)` for N-component PDE systems. |

---

## 6. Tier 5 — Battery P2D + PINN + GNN + advanced (~3 wk)

These are the lower-priority chapters of the User's Guide.  Each
needs its own micro-roadmap.

| Feature | Status | Notes |
|---|:-:|---|
| `batteryP2DModel` — pseudo-2-D lithium-ion battery model | 🔵 | Couples solid-phase / electrolyte / SEI layers; the equations are documented but the user-facing surface is small (`batteryP2DModel` + `solve`).  ~1 wk standalone. |
| Physics-Informed Neural Networks (`solvePoissonPINN` style) | 🔵 | Needs a tensor framework integration (PyTorch via `torch.utils.cpp_extension` or in-house autodiff) — out of immediate scope.  Carve out for now. |
| Graph Neural Networks for heat equation | 🔵 | Same — carve out. |
| Fourier Neural Operator for 3-D battery cooling | 🔵 | Same — carve out. |
| Nédélec edge elements (3-D magnetostatic curl-curl) | 🔵 | Necessary to ship the full 3-D magnetostatic / harmonic-EM path; ~1 wk. |
| STEP file import | 🔵 | Needs OpenCASCADE Community Edition or `oce`.  Carve out unless a user requests it explicitly. |
| `pdsolve` symbolic ↔ FEM bridge | 🟡 | Already partially shipped on the symbolic side; coupling them is mostly docs. |

---

## 7. Tier 6 — Apps + Live Editor + Simulink (carved out)

| Feature | Status | Notes |
|---|:-:|---|
| PDE Modeler 2-D app | 🔵 | Qt app — carve out (same precedent as Filter Designer, Signal Analyzer, RF Budget Analyzer). |
| Visualize PDE Results Live Editor Task | 🔵 | Same — carve out. |
| Simulink S-function for FEM models | 🔵 | Simulink not in scope (general repo posture). |

---

## 8. Compile / Execute pipeline

Same precedent as the SPT / CST / RF arcs:

1. **Sema** — `lib/Sema/Resolver.cpp` registers the new builtins:
   `createpde`, `femodel`, `materialProperties`, `faceBC`, `edgeBC`,
   `vertexBC`, `faceLoad`, `edgeLoad`, `vertexLoad`, `cellLoad`,
   `multicuboid`, `multicylinder`, `multisphere`, `importGeometry`,
   `geometryFromEdges`, `applyBoundaryCondition`,
   `specifyCoefficients`, `setInitialConditions`, `generateMesh`,
   `refineMesh`, `solvepde`, `solvepdeeig`, `solve` (method-dispatch
   on `femodel`), `interpolateSolution`, `interpolateDisplacement`,
   `interpolateStress`, `evaluateStrain`, `evaluateVonMisesStress`,
   `evaluatePrincipalStress`, `assembleFEMatrices`, `pdegplot`,
   `pdemesh`, `pdeplot`, `pdeplot3D`, `pdeviz`, `findNodes`,
   `findElements`.

2. **MLIR** — new pass `lib/MLIR/Passes/LowerPDE.cpp` that rewrites
   `matlab.call_builtin "pde_*"` into `llvm.call @matlab_pde_*`.
   Generic-PDE coefficient functions (the `@(location, state)`
   handles passed to `specifyCoefficients`) flow through the same
   anon-handle-retyping pre-pass already used by `ode45_v` —
   `LowerAnonCalls` retypes `location` / `state` block args from f64
   to ptr when the call site has struct arguments.

3. **Runtime** — `runtime/runtime_pde.cpp` + `runtime/runtime_pde_mesh.cpp`
   + `runtime/runtime_pde_stl.cpp` + `runtime/runtime_pde_assembly.cpp`,
   linked into `matlabc` so the JIT can resolve `matlab_pde_*` via
   the existing `DynamicLibrarySearchGenerator`.  Companion classdef
   sugar in `runtime/pde_classdefs.m` (same pattern as
   `runtime/cst_classdefs.m`).

4. **Codegen lanes** — Python and TypeScript get **only** the
   Tier-1 numeric core (mesher + assembly + sparse solve + 2-D
   `pdeplot`).  Tier-2 3-D + STL + `pdeplot3D` are LLVM-only at
   first cut, gated by `.skip-emit-python` / `.skip-emit-typescript`
   on the relevant Run/ tests, following the precedent set by
   complex-valued signal tests.

---

## 9. Debug / REPL

### 9.1 REPL renderer for new descriptor types

`runtime/runtime_debug.cpp` will get one renderer per new classdef:

```
>> R
R =
  StaticStructuralResults with properties:
    Displacement: <12 348 × 1 struct (ux, uy, uz, Magnitude)>
          Stress: <12 348 × 1 struct (xx, yy, zz, xy, yz, xz)>
          Strain: <12 348 × 1 struct (xx, yy, zz, xy, yz, xz)>
  VonMisesStress: [12348 × 1 double]
            Mesh: FEMesh (12 348 nodes, 23 117 tetrahedra)

>> R.VonMisesStress(1:5)
ans =
   1.4e+07
   2.1e+07
   ...
```

### 9.2 DAP variable inspector

Each new classdef gets a `dap_children(obj)` walker exposing the
struct-of-arrays fields as child nodes.  The pattern is exactly the
one used by `RFCktAmplifier` / `RFSparameters` today.

### 9.3 Debug breakpoints

The breakpoint surface is unchanged — `solve(model)` is a function
call like any other, so users can break inside their `@pdefun` /
`@bcfun` / `@(location, state) ...` handles via the existing
`-emit-debug` lane.  No additional work.

---

## 10. Cross-cutting prerequisites

These are not tier-specific but every tier depends on them.  They're
called out as separate sub-projects because each is a few sessions
of focused work on its own.

### 10.1 Sparse matrices (prerequisite for all of Tier 1+)

Without a sparse representation, the global stiffness matrix `K`
explodes the dense memory budget at ~1 000 DOF.  Required surface:

- `matlab_mat_sparse` descriptor kind in `runtime/runtime_internal.h`;
  CSR layout; `make_sparse(rows, cols, vals, m, n)` constructor.
- `sparse(...)`, `full(...)`, `nnz(...)`, `find(S)`, `spdiags`,
  `spones`, `speye`, `sprand`, `kron(A, B)` (sparse-aware).
- Arithmetic: `S + S`, `S - S`, `S * S`, `S * v`, `v' * S`, `S \ v`.
- Sparse-direct: in-tree column-pivoted LU (Davis textbook),
  symbolic + numeric phases.  ~500 LOC C.
- Iterative: `pcg`, `bicgstab`, `gmres`, `minres` (all on
  `(A, b, tol, maxit)`), with ILU(0) preconditioner.
- REPL display: `S` prints `[m × n sparse, nnz = …]`.

**Effort**: 1 wk.  Gates everything else here, so this lands first.

### 10.2 Unstructured-mesh plotting (prerequisite for Tier-1.6 + Tier-2.6)

- 2-D triangulated solution rendering: extend `runtime/plot/` with
  a `trisurf2d` painter that takes `(nodes, triangles, nodal_data,
  cmap)` and emits Gouraud-shaded Cairo polygons.  ~150 LOC.
- 3-D boundary-mesh rendering: `trisurf3d` painter; Painter's-
  algorithm depth sort over the boundary triangles, Lambertian
  shading, per-face colormap.  ~300 LOC.
- Deformed-shape rendering: per-node displacement vector applied
  before the depth sort; `DeformationScaleFactor` kwarg.
- Edge / face / vertex label overlays (`pdegplot` labels).
- Animation: write N PNGs into a directory; user assembles with
  `ffmpeg` externally (no interactive viewer in scope).

**Effort**: 1 wk.

### 10.3 Triangulation + tet-meshing (prerequisite for Tier-1.2 + Tier-2.2)

- 2-D constrained Delaunay — `distmesh`-style in-house (~150 LOC) or
  in-tree Ruppert's algorithm (~500 LOC).
- 3-D Delaunay-of-Bowyer-Watson + Lawson flips with octree-driven
  refinement (~800 LOC) — license-clean path.
- Opt-in TetGen integration via `MATLAB_LLVM_WITH_TETGEN=ON` for
  production-quality meshes.
- Quadratic-tet upgrade pass (linear → 10-node, adds mid-edge nodes).

**Effort**: 2 wk.

### 10.4 STL importer (prerequisite for "bring your own 3-D model" tier-2 row)

- ASCII + binary STL parsing into a triangulated surface mesh.
- Vertex welding via hash-by-quantized-coordinate.
- Region inference (flood-fill on connected-face graph) for
  `fegeometry` multi-cell support.

**Effort**: 2 sessions.  Lives in `runtime/runtime_pde_stl.cpp`.

### 10.5 Eigenvalue solver upgrades

- Generalised symmetric eig `K v = λ M v` — Krylov-Schur / Lanczos
  with shift-invert.  Required for `solvepdeeig` and
  `structuralModal`.  ARPACK-style (~600 LOC); reuses the existing
  `lu_decompose` for the shift-invert step.

**Effort**: 1 wk.

---

## 11. Carved out

- Simulink S-function for FEM models (Simulink out of scope).
- PDE Modeler 2-D app (Qt app; same precedent as Signal Analyzer).
- Visualize PDE Results Live Editor Task.
- STEP file import (needs OpenCASCADE).
- Physics-Informed Neural Networks (`solvePoissonPINN`-style) — needs
  a tensor/autodiff framework integration.
- Graph Neural Networks / Fourier Neural Operator for heat / battery
  — same.
- Battery P2D modeling (Tier-5; not on the user's critical path).
- 3-D Nédélec edge elements for full 3-D vector magnetostatic / HF
  electromagnetics.
- Interactive viewer for `pdeviz` animations — we write PNG frames
  and let `ffmpeg` assemble.

---

## 12. Execution order

Tight critical path to the **headline wind-stress demo**:

| Order | Item | Effort | Why this order |
|:-:|---|:-:|---|
| 1 | §10.1 Sparse matrices | 1 wk | Everything else needs it. |
| 2 | §10.3 2-D triangulation | 1 wk | Tier-1 needs it; gates Tier-2's 3-D mesher. |
| 3 | §10.3 3-D tet mesher | 1 wk | Tier-2 critical. |
| 4 | §10.4 STL importer | 2 sess | Tier-2 critical (user wants their own model). |
| 5 | §10.2 Unstructured-mesh plotting (2-D + 3-D) | 1 wk | Tier-1 and Tier-2 critical. |
| 6 | §2 Tier-1 — 2-D elliptic loop (`createpde` / `solvepde` / `pdeplot`) | 1 wk | Validates assembly + sparse solve on the simpler 2-D case before 3-D. |
| 7 | §3 Tier-2 — 3-D unified `femodel` workflow + structural-static + the headline wind-stress demo | 1.5 wk | **Done-bar.** |
| 8 | §10.5 Eigenvalue upgrade | 1 wk | Unblocks modal / frequency response. |
| 9 | §3.3 + §3.4 Tier-2 finalisation — modal + tuning-fork sanity check | 1 wk | Closes the Tier-2 example tail. |
| 10 | §4 Tier-3 — transient + thermal + EM | 3 wk | Broadens the User's-Guide example coverage. |
| 11 | §5 Tier-4 — ROM + nonlinear + adaptive | 2 wk | Wing-spar Craig-Bampton + clamped-beam nonlinear. |
| 12 | §6 Tier-5 — battery P2D + remaining specialty (if any user asks) | 3 wk | Lowest priority. |

**Total critical-path-to-demo: ~8 wk** (sparse → mesher → STL →
plotting → Tier-1 sanity → Tier-2 + demo).  Two-thirds of that is
the prerequisite infrastructure; the actual PDE-specific assembly
+ load + solve code is only ~3 wk.

---

## 13. Open questions

1. **Mesher licence posture**: ship the in-tree Bowyer-Watson 3-D
   mesher by default (license-clean, slower) and gate TetGen behind
   `MATLAB_LLVM_WITH_TETGEN=ON` (BSD-with-attribution but adds a 14k
   LOC C++ dependency)?  Recommended yes — same precedent as
   `MATLAB_LLVM_WITH_SYM` for SymPP.

2. **Sparse-direct vs. iterative default**: dense problems below
   ~5 k DOF run faster on a direct LU; iterative wins above ~50 k.
   Recommend: `solvepde` auto-selects by DOF count, override via
   `solvepde(model, SolverHints=struct("LinearSolver","direct"))`.

3. **Python / TypeScript lane coverage**: which Tier-2 features get
   `.skip-emit-python` / `.skip-emit-typescript`?  Suggested: all
   3-D plotting (`pdeplot3D`, `pdeviz`); the numeric solve path is
   portable.

4. **DAP renderer richness**: do `R.Stress.xx` etc. get child-walked
   in the DAP tree, or kept as flat structs?  Recommended:
   child-walked, with a max-depth cap of 3 (matches RF precedent).
