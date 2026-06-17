# Symbolic Math Toolbox Spec

## Purpose
Documents the shipped subset of the Symbolic Math Toolbox in the matlab_llvm compiler: a MATLAB-compatible computer-algebra front end backed by SymPP (a C++ port of SymPy), opt-in via the CMake flag `-DMATLAB_LLVM_WITH_SYM=ON` (links SymPP + GMP/MPFR). Symbolic values are opaque runtime types (kind=7 sym, kind=8 symmat) that persist across REPL inputs. Tiers 1-4 are marked shipped. (doc: docs/symbolic_toolbox_roadmap.md) (doc: docs/sym.md) (src: runtime/toolbox/sym)

## Requirements

### Requirement: Opt-in SymPP build integration
The system SHALL gate all symbolic functionality behind an opt-in build that links SymPP. (doc: docs/sym.md) (src: runtime/toolbox/sym/runtime_sym_stub.cpp)

#### Scenario: Build with and without the symbolic backend
- **WHEN** the project is configured with `-DMATLAB_LLVM_WITH_SYM=ON` (with SymPP + GMP/MPFR available) versus the default OFF
- **THEN** the system SHALL link the full `matlab_sym_*` runtime when ON, and provide no-op stubs (sym tests skip) when OFF

### Requirement: Symbolic creation, arithmetic, and elementary functions
The system SHALL provide symbol creation, operator overloads, and elementary functions. (src: runtime/toolbox/sym/runtime_sym.cpp)

#### Scenario: Create and combine symbols
- **WHEN** a program calls `syms`/`sym`/`str2sym` and combines symbols with `+ - * / ^ ==` or elementary functions (`sin`/`cos`/`tan`/`exp`/`log`/`sqrt`/`abs` and inverses/hyperbolics)
- **THEN** the system SHALL produce a symbolic (kind=7) expression that renders via `disp`/`latex`/`pretty`/`ccode`

### Requirement: Algebra, simplification, and equation solving
The system SHALL provide simplification and symbolic equation solving. (src: runtime/toolbox/sym/runtime_sym.cpp)

#### Scenario: Simplify and solve
- **WHEN** a program calls `simplify`, `expand`, `factor`, `subs`, `solve`/`solve_one`, or numeric evaluation `double`/`vpa`
- **THEN** the system SHALL return the simplified/factored expression, the substitution, the symbolic root array, or the numeric value

### Requirement: Calculus, transforms, and differential equations
The system SHALL provide calculus, integral transforms, and ODE/PDE solvers. (src: runtime/toolbox/sym/runtime_sym.cpp)

#### Scenario: Differentiate, transform, and solve ODEs
- **WHEN** a program calls `diff`, `int`, `taylor`, `limit`, transforms (`laplace`/`ilaplace`, `fourier`/`ifourier`, `ztrans`/`iztrans`), or `dsolve`/`dsolve_ivp`/`pdsolve`/`pdsolve_heat`/`pdsolve_wave`/`checkodesol`
- **THEN** the system SHALL return the derivative/integral/series/limit, the transform pair, or the (initial-value) differential-equation solution

### Requirement: Symbolic matrices and multi-equation solvers
The system SHALL provide symbolic matrix algebra and multi-equation solving. (src: runtime/toolbox/sym/runtime_sym.cpp)

#### Scenario: Symbolic linear algebra
- **WHEN** a program builds a symbolic matrix (kind=8) via literals/`sym_matrix`/`sym_eye`/`sym_zeros` and calls `inv`/`det`/`trace`/`rank`/`eig`/`chol`/`lu`/`qr`/`linsolve`, or solves a system via `sym_solve_sys`/`sym_solve_2x2`/`sym_solve_3x3`
- **THEN** the system SHALL return the closed-form matrix result or the joint solution set

### Requirement: Assumptions and numeric solvers
The system SHALL provide an assumptions framework and arbitrary-precision numeric solvers. (src: runtime/toolbox/sym/runtime_sym.cpp)

#### Scenario: Apply assumptions and solve numerically
- **WHEN** a program calls `assume`/`assumeAlso`/`clearAssumptions` (real, integer, positive, etc.) or `nsolve`/`vpasolve`
- **THEN** the system SHALL propagate the assumption mask through `simplify`/`refine` and return MPFR-precision roots
