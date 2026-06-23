## ADDED Requirements

### Requirement: Variable-step explicit solvers

The mflowLink simulator SHALL provide variable-step explicit integrators selected by
`settings.solver.algorithm`: `ode45` (Dormand–Prince 5(4)) and `ode23` (Bogacki–Shampine
3(2)), each with an embedded lower-order error estimate driving a shared step-size
controller that honours `relTol`, `absTol`, `maxStep`, and `minStep`. `ode23` SHALL use the
native Bogacki–Shampine 3(2) pair, not an alias of the 5(4) method.

#### Scenario: Adaptive step honours tolerance
- **WHEN** a smooth non-stiff model is simulated with `type: variable_step`, `algorithm: ode45`
- **THEN** the solver grows and shrinks the step from the embedded error estimate, keeping the
  per-step error norm within the configured `relTol`/`absTol`, and never exceeds `maxStep`

#### Scenario: ode23 is the native 3(2) pair
- **WHEN** a model selects `algorithm: ode23`
- **THEN** the integrator advances with the Bogacki–Shampine 3(2) tableau (third-order step,
  embedded second-order estimate) and agrees with `ode45` to tolerance on a smooth reference

### Requirement: Variable-order, variable-step stiff BDF (ode15s)

The simulator SHALL provide `ode15s` as a variable-order (orders 1–5), variable-step implicit
BDF integrator with a Newton corrector, a Jacobian (finite-difference by default), and a dense
linear solve. Order and step SHALL be selected from the local error estimate and Newton
convergence; the solver history SHALL reset to order 1 on any discrete-state change,
zero-crossing reset, or solver restart. The order-1, fixed-step configuration (Backward Euler)
SHALL remain reachable and numerically unchanged.

#### Scenario: Stiff system without tiny steps
- **WHEN** a stiff dissipative system is simulated with `algorithm: ode15s`
- **THEN** the solver remains stable and accurate using steps far larger than an explicit
  method's stability limit, raising the BDF order while Newton converges cleanly

#### Scenario: Backward-Euler compatibility
- **WHEN** a model pins `ode15s` to order 1 with a fixed step
- **THEN** the output matches the prior BDF1 behaviour

### Requirement: Additional stiff and moderately-stiff solvers

The simulator SHALL recognise and implement `ode23s` (Rosenbrock-W, one-step linearly
implicit), `ode23t` (trapezoidal, non-dissipative), and `ode23tb` (TR-BDF2). These names
SHALL NOT silently fall through to a fixed-step explicit method.

#### Scenario: Rosenbrock stiff step
- **WHEN** a stiff model selects `algorithm: ode23s`
- **THEN** the solver advances with the Rosenbrock-W stages using one Jacobian evaluation per
  step (no Newton iteration loop) and remains stable on the stiff system

#### Scenario: Unimplemented-alias removed
- **WHEN** a model selects `ode23s`, `ode23t`, or `ode23tb`
- **THEN** the simulator uses the named method rather than the classic RK4 fallback

### Requirement: Jacobian reuse and analytic Jacobian hook

The implicit solvers SHALL amortise the Jacobian and its factorisation across Newton
iterations and across steps, refactoring only when the step size changes beyond a threshold,
Newton convergence stalls, or the BDF order changes. The solver SHALL accept an optional
model-supplied analytic Jacobian; absent one, it SHALL use a forward-difference Jacobian.

#### Scenario: Jacobian held while Newton converges
- **WHEN** an implicit solver takes several steps at a stable step size with converging Newton
  iterations
- **THEN** the Jacobian is reused (not recomputed every iteration) and is refactored only when
  the step, order, or convergence rate forces it

### Requirement: Mass matrix and index-1 DAE

The implicit lane SHALL support a mass matrix `M·(y − y_old) − h·f`, with Jacobian
`M − h·∂f/∂y`. A constant `M` SHALL be configurable; a singular `M` (semi-explicit index-1
DAE) SHALL be permitted. An explicit solver selected together with a non-identity mass matrix
SHALL raise a sourced error.

#### Scenario: Singular mass matrix solved implicitly
- **WHEN** a semi-explicit index-1 DAE with a singular constant `M` is simulated with `ode15s`
- **THEN** the implicit step solves `M − h·∂f/∂y` (non-singular for small `h`) and integrates
  the system without inverting `M` alone

#### Scenario: Mass matrix rejected on an explicit solver
- **WHEN** a non-identity mass matrix is paired with `ode45` or `ode23`
- **THEN** lowering reports a sourced error directing the user to an implicit method

### Requirement: Dense output and Refine

The variable-step solvers SHALL provide a continuous interpolant over each accepted step
(DOPRI5 dense output for the explicit lane; the BDF history polynomial for the implicit lane).
A `settings.solver.refine = k` SHALL emit `k` evenly-spaced interpolated samples per accepted
step; `refine = 1` (endpoints only) SHALL keep logged output byte-identical to the
pre-change behaviour, and zero-crossing localisation SHALL use the interpolant.

#### Scenario: Refine densifies the log without shrinking the step
- **WHEN** a model sets `refine: 4` with a variable-step solver
- **THEN** the trace contains four interpolated samples per accepted step while the solver's
  step-size control is unchanged

#### Scenario: Default refine is byte-identical
- **WHEN** a model leaves `refine` unset (or `refine: 1`)
- **THEN** the logged samples are exactly the accepted step endpoints, as before

### Requirement: Interpreter and compiled solver parity

Every solver method SHALL produce byte-identical output in the in-process interpreter
(`matlabc -simulate`) and the standalone `matlabc -emit-mflowlink-cpp` binary, since both link
the same `MflowLinkSim` evaluator.

#### Scenario: Compiled stiff run matches the interpreter
- **WHEN** a stiff or variable-step model is run via `-simulate` and via the compiled
  `-emit-mflowlink-cpp` binary
- **THEN** the two CSV traces are byte-identical
