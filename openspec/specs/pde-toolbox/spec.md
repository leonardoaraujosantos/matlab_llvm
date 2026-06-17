# Partial Differential Equation Toolbox Spec

## Purpose
Documents the shipped subset of the PDE Toolbox in the matlab_llvm compiler: a Finite Element Method pipeline for 2-D/3-D linear and nonlinear PDEs via the `femodel` workflow, spanning model creation, geometry/meshing, boundary conditions, sparse assembly, Krylov and eigenvalue solvers, time-stepping, and structural/thermal/electromagnetic post-processing. Tiers 0-4 are marked shipped (2026-05-13). (doc: docs/pde_toolbox_roadmap.md) (src: runtime/toolbox/pde)

## Requirements

### Requirement: Model creation and setup
The system SHALL provide a unified PDE model container with materials, boundary conditions, and loads. (src: runtime/toolbox/pde/pde_classdefs.m) (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Build a model
- **WHEN** a program calls `femodel(AnalysisType, Geometry, ...)` (or legacy `createpde`) and sets `materialProperties`, boundary value types (`faceBC`/`edgeBC`/`vertexBC`), and loads (`faceLoad`/`edgeLoad`/`cellLoad`) via the `pde_set_*` setters
- **THEN** the system SHALL return a model dispatchable on AnalysisType (structuralStatic/Transient/Modal/Frequency, thermalSteadyState/Transient, electrostatic/magnetostatic/dcConduction/harmonicElectromagnetic, structuralStaticNL/TL)

### Requirement: Geometry and meshing
The system SHALL provide structured and voxelized mesh generation, import, refinement, and transforms. (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Generate or import a mesh
- **WHEN** a program calls mesh generators (`matlab_pde_mesh_cuboid_tet`, `matlab_pde_multicylinder`/`_hollow`, `matlab_pde_multisphere`, `matlab_pde_voxelize_surface`, `matlab_pde_mesh_quadratic`), importers (`matlab_pde_load_stl`, `matlab_pde_load_glb`), or refinement/transform functions (`matlab_pde_refine_mesh`, `matlab_pde_refine_mesh_bey`, `matlab_pde_adapt_mesh`, translate/rotate/scale)
- **THEN** the system SHALL return a tetrahedral mesh (T4 or quadratic T10) with node/face accessors

### Requirement: FEM assembly and sparse linear solve
The system SHALL provide FEM assembly for Poisson and elasticity with Dirichlet enforcement and Krylov solvers. (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Assemble and solve a linear system
- **WHEN** a program calls assembly (`matlab_pde_assemble_poisson_2d`/`_3d_sparse`, `matlab_pde_assemble_elast_3d_sparse`), applies Dirichlet/fixed constraints, and solves with `matlab_sparse_pcg`, `matlab_sparse_gmres_ilu0`, or `matlab_sparse_minres`
- **THEN** the system SHALL return the solution vector with iteration/residual diagnostics from CSR sparse infrastructure

### Requirement: Analysis-type solve dispatch
The system SHALL provide a unified solver dispatching on AnalysisType across physics domains. (doc: docs/pde_toolbox_roadmap.md) (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Solve structural, thermal, or electromagnetic problems
- **WHEN** a program calls `solve`/`matlab_pde_solve_femodel` (or legacy `solvepde`) on a model
- **THEN** the system SHALL run the matching kernel (static/transient/modal/frequency elasticity, steady/transient thermal with k(T) Picard, electrostatic/magnetostatic/dcConduction/harmonicEM) and return a result object (e.g. `StaticStructuralResults`, `ThermalResults`, `ElectrostaticResults`)

### Requirement: Eigenvalue and reduced-order analysis
The system SHALL provide modal eigensolvers and reduced-order modeling. (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Compute modes and reduce a model
- **WHEN** a program calls eigensolvers (`matlab_pde_eigsmall`, `matlab_pde_eig_lanczos_si`/`_full`) or ROM functions (`matlab_pde_reduce`, `matlab_pde_reduce_craig_bampton`, `matlab_pde_reconstruct_solution`)
- **THEN** the system SHALL return natural frequencies and mode shapes, or a reduced model and reconstructed full-field solution

### Requirement: Post-processing and visualization
The system SHALL provide stress recovery and mesh/field plotting. (src: runtime/toolbox/pde/runtime_pde.cpp)

#### Scenario: Recover stress and plot results
- **WHEN** a program calls stress recovery (`matlab_pde_von_mises_3d`, `matlab_pde_node_von_mises_3d`, `matlab_pde_peak_disp_3d`) or plotting (`pdeplot`, `matlab_pdeplot3d`, `matlab_pdeplot3d_deformation`)
- **THEN** the system SHALL return per-element/nodal von Mises stress and peak displacement, or render the mesh/deformed-field image
