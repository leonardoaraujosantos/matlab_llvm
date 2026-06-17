# ODE/PDE Solvers Spec

## Purpose
Documents the observed behavior of the differential-equation solvers in the runtime: explicit ODE integrators (ode45, ode23), the stiff Rosenbrock solver (ode23s), event detection, the 1-D PDE solver (pdepe), honored odeset options, and the solver output formats. Records what is implemented today versus what remains on the DAE roadmap (src: runtime/matlab_runtime.cpp; doc: docs/ode.md, docs/dae_solver_roadmap.md).

## Requirements

### Requirement: Explicit ODE solvers
The system SHALL provide ode45 (Dormand-Prince 5(4)) and ode23 (Bogacki-Shampine 3(2)) for non-stiff scalar and vector initial-value problems.

#### Scenario: Integrate a non-stiff system
- **WHEN** a program calls `ode45` or `ode23` with an RHS handle, time span, and initial condition (scalar or vector state)
- **THEN** the system SHALL return the time grid and state trajectory using adaptive step-size control with dense-output refinement (src: runtime/matlab_runtime.cpp matlab_ode45_t/matlab_ode45_v_y/matlab_ode23_t; test: test/Run/math_ode45_basic.m)

### Requirement: Stiff ODE solver
The system SHALL provide ode23s, a Rosenbrock W-method solver for stiff systems, using a finite-difference Jacobian per step.

#### Scenario: Integrate a stiff system
- **WHEN** a program calls `ode23s` on a stiff problem (scalar or vector)
- **THEN** the system SHALL integrate using the `(I - h·d·J)` implicit factorization with a central-difference Jacobian and return the time/state trajectory (src: runtime/matlab_runtime.cpp matlab_ode23s_t/matlab_ode23s_v_y; test: test/Run/math_ode23s_basic.m, test/Run/math_ode23s_robertson.m)

### Requirement: Event detection
The system SHALL provide a dedicated event-detection builtin that locates zero crossings of an event function during integration via bracket-then-bisect and returns terminal/direction-aware crossings.

#### Scenario: Detect a terminal event
- **WHEN** a program calls the event-aware ODE builtin with an event function returning `[value; isterminal; direction]`
- **THEN** the system SHALL return the standard trajectory plus the event times, states, and indices, bisecting (up to 50 iterations, tolerance `1e-12`) within the accepted step (src: runtime/matlab_runtime.cpp matlab_ode_events_t/matlab_ode_events_te/matlab_ode_events_ie; doc: docs/ode.md; test: test/Run/math_ode_events_ball.m)

### Requirement: 1-D PDE solver (pdepe)
The system SHALL provide pdepe for a single 1-D parabolic/elliptic PDE on a user mesh, supporting Cartesian/cylindrical/spherical symmetry and Dirichlet/Neumann/Robin boundary conditions.

#### Scenario: Solve a 1-D heat equation
- **WHEN** a program calls `pdepe(m, pdefn, icfn, bcfn, xmesh, tspan)` for a scalar PDE
- **THEN** the system SHALL discretize by method-of-lines, integrate the resulting ODE system with the stiff solver, and return an `N_t × N_x` solution array (src: runtime/matlab_runtime.cpp matlab_pdepe; doc: docs/ode.md; test: test/Run/math_pdepe_heat.m, test/Run/math_pdepe_neumann.m, test/Run/math_pdepe_radial.m)

#### Scenario: Symmetry coordinate constraint
- **WHEN** a program calls `pdepe` with `m = 1` or `m = 2` (cylindrical/spherical)
- **THEN** the system SHALL require `xmesh(1) > 0` because the axis-of-symmetry singularity at `xmesh(1) = 0` is deferred (src: runtime/matlab_runtime.cpp matlab_pdepe; doc: docs/ode.md)

### Requirement: Solver options (odeset)
The system SHALL honor a defined subset of odeset fields and SHALL silently ignore the rest.

#### Scenario: Apply tolerances and step controls
- **WHEN** a program passes an options struct with `RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, or `Stats`
- **THEN** the system SHALL apply those fields (defaults: RelTol 1e-3, AbsTol 1e-6, Refine 4 for ode45 / 1 for ode23/ode23s) and SHALL ignore unsupported fields such as `Events`, `OutputFcn`, `Jacobian`, and `Mass` (src: runtime/matlab_runtime.cpp ode_opts_resolve; doc: docs/ode.md)

### Requirement: Solver output formats
The system SHALL return solver results as a time grid plus a state array, with an optional statistics struct, and dense interpolation between accepted steps.

#### Scenario: Request trajectory and statistics
- **WHEN** a program calls a solver in `[t, y]` or `[t, y, stats]` form
- **THEN** the system SHALL return `t` as an N×1 column, `y` as N×1 (scalar) or N×D (vector) with `y(i,:)` the state at `t(i)`, and a stats struct with `nsteps`/`nfailed`/`nfevals` when requested (src: runtime/matlab_runtime.cpp matlab_ode45_stats/ode_stats_struct_from_cache; doc: docs/ode.md)

### Requirement: DAE support is roadmap-only
The system SHALL NOT currently provide mass-matrix or fully implicit DAE solvers (ode15s, ode15i) or higher-order solvers (ode113, ode78, ode89); these remain planned work.

#### Scenario: Mass-matrix solver unavailable
- **WHEN** a program attempts to use ode15s/ode15i or relies on the `Mass` odeset field
- **THEN** the system SHALL not resolve a runtime implementation, as these are documented as not started (src: doc: docs/dae_solver_roadmap.md; doc: docs/ode.md)
