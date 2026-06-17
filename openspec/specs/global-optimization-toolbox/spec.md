# Global Optimization Toolbox Spec

## Purpose
Documents the shipped subset of the Global Optimization Toolbox in the matlab_llvm compiler: derivative-free and stochastic global solvers for multi-modal, noisy, or discontinuous objectives, with optional local refinement via the shipped Optimization Toolbox. All six tiers are marked shipped (2026-05-20). (doc: docs/global_optim_toolbox_roadmap.md) (src: runtime/toolbox/gads)

## Requirements

### Requirement: Single-objective stochastic solvers
The system SHALL provide population-based and annealing global solvers. (src: runtime/toolbox/gads/runtime_gads.cpp)

#### Scenario: Stochastic global minimization
- **WHEN** a program calls `ga`, `particleswarm`, or `simulannealbnd`
- **THEN** the system SHALL return a minimizer computed by a real-coded genetic algorithm, Clerc-Kennedy constriction PSO, or bounded simulated annealing respectively

### Requirement: Multi-start global meta-solvers
The system SHALL provide multi-start meta-solvers that orchestrate local solves. (doc: docs/global_optim_toolbox_roadmap.md) (src: runtime/toolbox/gads/gads_classdefs.m)

#### Scenario: Multi-start refinement
- **WHEN** a program builds a problem with `createOptimProblem` and calls `run` on a `MultiStart` or `GlobalSearch` object
- **THEN** the system SHALL launch multiple random-start `fmincon` local solves and return the best solution found

### Requirement: Deterministic direct search
The system SHALL provide a deterministic pattern-search solver. (src: runtime/toolbox/gads/runtime_gads.cpp)

#### Scenario: Pattern search on a nonsmooth objective
- **WHEN** a program calls `patternsearch`
- **THEN** the system SHALL return a minimizer via Generalized Pattern Search with a complete plus/minus basis poll and adaptive mesh, with no PRNG

### Requirement: Surrogate-based optimization
The system SHALL provide a surrogate-based global solver for expensive objectives. (src: runtime/toolbox/gads/runtime_gads.cpp)

#### Scenario: Surrogate optimization
- **WHEN** a program calls `surrogateopt`
- **THEN** the system SHALL fit a cubic RBF surrogate, sample adaptively with merit-weighted candidate selection, and return the best point after a final `fmincon` polish

### Requirement: Multiobjective Pareto solvers
The system SHALL provide multiobjective solvers returning a Pareto set. (src: runtime/toolbox/gads/runtime_gads.cpp)

#### Scenario: Compute a Pareto front
- **WHEN** a program calls `gamultiobj` or `paretosearch`
- **THEN** the system SHALL return a non-dominated set computed by NSGA-II (fast non-dominated sort + crowding distance) or a GPS-poll archive with crowding-distance pruning respectively

### Requirement: Solver options and integer constraints
The system SHALL provide an options carrier and integer-constrained genetic optimization. (src: runtime/toolbox/gads/runtime_gads.cpp) (src: runtime/toolbox/gads/gads_classdefs.m)

#### Scenario: Mixed-integer GA via options
- **WHEN** a program builds `optimoptions('ga', ...)` with `PopulationSize`/`MaxGenerations`/`IntCon` and calls `ga`
- **THEN** the system SHALL honor the options and round integer-constrained variables at initialization, crossover, mutation, and result
