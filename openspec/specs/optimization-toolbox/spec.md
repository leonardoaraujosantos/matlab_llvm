# Optimization Toolbox Spec

## Purpose
Documents the shipped subset of the Optimization Toolbox in the matlab_llvm compiler: hand-coded numerical solvers (no external dependency) for unconstrained and constrained nonlinear optimization, linear/quadratic/mixed-integer programming, nonlinear least-squares, and equation solving, plus a problem-based expression-DAG API. All five tiers are marked shipped (2026-05-14). (doc: docs/optim_toolbox_roadmap.md) (src: runtime/toolbox/optim)

## Requirements

### Requirement: Unconstrained solvers and root finding
The system SHALL provide root finding and unconstrained minimization solvers in function form. (src: runtime/toolbox/optim/runtime_optim.cpp)

#### Scenario: Scalar root and 1-D minimization
- **WHEN** a program calls `fzero`, `fminbnd`, `fminsearch`, or `fminunc`
- **THEN** the system SHALL return a minimizer/root computed by Brent bracket-search, golden-section localmin, Nelder-Mead simplex, or BFGS quasi-Newton with finite-difference gradients respectively

### Requirement: Constrained nonlinear and least-squares solvers
The system SHALL provide constrained nonlinear optimization and least-squares solvers. (doc: docs/optim_toolbox_roadmap.md) (src: runtime/toolbox/optim/runtime_optim.cpp)

#### Scenario: Constrained minimization and curve fitting
- **WHEN** a program calls `fmincon`, `quadprog`, `lsqlin`, `lsqnonlin`, `lsqcurvefit`, `lsqnonneg`, or `fsolve`
- **THEN** the system SHALL return a solution computed by the augmented-Lagrangian core (fmincon/quadprog/lsqlin), Levenberg-Marquardt with bounds (lsqnonlin/lsqcurvefit/fsolve N-D), or Lawson-Hanson NNLS (lsqnonneg)

### Requirement: Linear, mixed-integer, and cone programming
The system SHALL provide linear, mixed-integer, cone, minimax, goal-attainment, and semi-infinite solvers. (src: runtime/toolbox/optim/runtime_optim.cpp)

#### Scenario: LP, MILP and specialized programs
- **WHEN** a program calls `linprog`, `intlinprog`, `coneprog`, `fminimax`, `fgoalattain`, or `fseminf`
- **THEN** the system SHALL return a solution via dense 2-phase simplex (linprog), depth-first branch-and-bound (intlinprog), second-order-cone reformulation (coneprog), epigraph reformulation (fminimax/fgoalattain), or sampled outer-approximation (fseminf)

### Requirement: Problem-based expression API
The system SHALL provide a problem-based API with optimization variables, expressions with operator overloads, and problem objects. (src: runtime/toolbox/optim/optim_classdefs.m)

#### Scenario: Build and solve a problem object
- **WHEN** a program constructs `optimvar`/`optimintvar` variables, builds an `OptimizationProblem` via `optimproblem`, and calls `solve`
- **THEN** the system SHALL evaluate the operator-overloaded expression DAG (`+`,`-`,`*`,`/`,`^`, `<=`,`>=`,`==`) and dispatch to the linear/QP/nonlinear backend, returning the solution

### Requirement: Problem-based equation solving
The system SHALL provide problem-based equation solving via equation problem objects. (doc: docs/optim_toolbox_roadmap.md) (src: runtime/toolbox/optim/optim_classdefs.m)

#### Scenario: Solve an equation system
- **WHEN** a program builds an `EquationProblem` via `eqnproblem`, assigns `Equations`, and calls `solve`
- **THEN** the system SHALL solve `F(x)=0` over the residual DAG by Levenberg-Marquardt and return the root
