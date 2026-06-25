## ADDED Requirements

### Requirement: sim3d package-dotted name folding

The parser SHALL fold the `sim3d` package-dotted references `sim3d.World`,
`sim3d.Actor`, and `sim3d.export` into flat names (`sim3d_World`, `sim3d_Actor`,
`sim3d_export`) at parse time, following the established package-fold convention
already used for `dsp.*`, `coder.gpu.*`, and `gpuArray.*` (lib/Parse/Parser.cpp
`parsePostfix`). The fold SHALL apply only when the leftmost segment is the bare
identifier `sim3d` followed by a recognised member; field access on a bound
variable named `sim3d` SHALL be unaffected.

#### Scenario: Qualified constructor folds to a flat class name

- **WHEN** code evaluates `w = sim3d.World();`
- **THEN** the parser produces a call to the flat class `sim3d_World`, which resolves to the prelude classdef `classdef sim3d_World < handle`

#### Scenario: Qualified function folds

- **WHEN** code evaluates `sim3d.export(w, 'scene.html')`
- **THEN** the parser folds it to `sim3d_export(w, 'scene.html')`, dispatching to the packaged export function

#### Scenario: Variable field access is unaffected

- **WHEN** a struct `sim3d` with field `World` exists and code evaluates `sim3d.World`
- **THEN** the fold does not apply to a member that is not a recognised sim3d entry, preserving normal field access semantics

### Requirement: Object-runtime calls forward the receiver to a C runtime

A classdef method body SHALL be able to invoke a registered `matlab_sim3d_*`
runtime entry, passing the receiver object (and matrix/scalar arguments) to a
C++ runtime that reads and writes the object's properties. The runtime symbol
SHALL be registered in the Sema builtin table and given an LLVM signature in the
object-runtime lowering table (receiver and matrices as pointer, scalars as
f64), mirroring the existing `matlab_dsp_*` System-Object convention.

#### Scenario: A Dependent property setter forwards a vector to the runtime

- **WHEN** an `Actor` declares `Translation` as a `Dependent` property with `set.Translation(obj, v)` calling `matlab_sim3d_set_translation(obj, v)`
- **THEN** `a.Translation = [x y z]` dispatches to the setter and the full vector reaches the runtime, which stores it against the object
