## ADDED Requirements

### Requirement: Device-resident gpuArray representation
The system SHALL represent the result of `gpuArray(X)` as a value the lowering
recognizes as device-resident and can route per-operation, distinct from a plain
host matrix. The representation SHALL round-trip through the REPL workspace (a value
assigned to a workspace variable and read back on a later turn SHALL still be
recognized as device-resident). When no real device backend is linked, the
representation SHALL remain backed by host data so results stay numerically correct
(the Tier-A fallback). This SHALL supersede the #333 identity-builtin behavior while
preserving the host correctness it secured.

#### Scenario: gpuArray value is tagged and host-correct
- **WHEN** a program runs `Ag = gpuArray(A); Cg = Ag * Bg; C = gather(Cg)` on a build
  with no device backend
- **THEN** `Cg` is carried as a device-resident value, `gather` returns a plain host
  matrix, and `C` equals the host `A * B` result (no empty/zero result)

#### Scenario: round-trips through the REPL workspace
- **WHEN** `Ag = gpuArray(A)` is entered on one REPL turn and `Ag` is used on a later turn
- **THEN** the later turn still treats `Ag` as device-resident (the tag is not lost
  across the workspace store/load)

### Requirement: Per-operation dispatch hook
The system SHALL route each supported operation whose operand(s) are device-resident
through a single dispatch point that selects the device backend when present and the
host CPU implementation otherwise, rather than calling the host runtime directly. In
Tier A the dispatch hook SHALL exist and always take the CPU-fallback path (no device
backend yet), producing results identical to the plain host path.

#### Scenario: op on a gpuArray goes through the dispatch hook
- **WHEN** an arithmetic or reduction op is applied to a device-resident operand
- **THEN** the op is emitted as a dispatched call (host fallback in Tier A) and not as
  a direct unconditional host-runtime call

#### Scenario: mixed host/device operands
- **WHEN** a binary op has one device-resident operand and one host scalar/matrix
- **THEN** the system SHALL dispatch the op as device-resident (host operand promoted),
  matching MATLAB's rule that a `gpuArray` operand makes the result a `gpuArray`

### Requirement: Uniform across compiled and interpreted lanes
The dispatch representation and hook SHALL behave identically on the AOT
(`-emit-llvm`/`-emit-c`/`-emit-cpp`) lanes and the in-process JIT pipeline
(`runJitSoftwareLowering`, used by Run/`-dap`/`-repl`), so a gpuArray program produces
the same results whether compiled or interpreted.

#### Scenario: same result compiled and interpreted
- **WHEN** the same gpuArray program is executed via `-emit-llvm` + link and via the
  JIT/REPL lane
- **THEN** both produce identical `gather`ed results

### Requirement: Device selection and fallback (later tiers)
When a real device backend is linked (Tier C), `MATLAB_GPU_TARGET=auto` SHALL escalate
to the present device (CUDA/Metal/OpenCL) and execute dispatched ops on it, with
`gpuArray(X)` performing a host→device upload and `gather` a device→host copy; when no
device is present the system SHALL fall back to the host CPU and stay numerically
correct. (Tier A ships only the always-fallback path; this requirement governs the
device path delivered in Tiers C–E.)

#### Scenario: device present
- **WHEN** a capable device is present and `MATLAB_GPU_TARGET=auto`
- **THEN** dispatched ops run on the device and `gather` returns host-correct data

#### Scenario: no device present
- **WHEN** no capable device is present
- **THEN** dispatched ops run on the host CPU and results are numerically correct
