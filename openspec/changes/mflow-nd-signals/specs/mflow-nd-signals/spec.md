## ADDED Requirements

### Requirement: N-D wire-signal shape

A mflowLink wire SHALL carry a shape of rank 1 to 6, as an ordered dimension list
`OutShape`, over a flat row-major buffer whose element count equals `prod(OutShape)`.
Element at multi-index `(i1,…,iN)` SHALL live at the row-major flat index
`((…(i1·d2 + i2)·d3 + i3)…)·dN + iN`. A declared or reshaped shape of rank > 6 SHALL be a
sourced error.

#### Scenario: Rank-3 element count
- **WHEN** a block declares an output shape `[2, 3, 4]`
- **THEN** the signal width is 24 and its shape metadata is `[2, 3, 4]`

#### Scenario: Rank over 6 rejected
- **WHEN** a model declares or reshapes a signal to rank 7 or higher
- **THEN** lowering reports a sourced error rather than silently truncating the rank

### Requirement: 1-D and 2-D backward compatibility

The N-D model SHALL be backward compatible: a width-`W` 1-D signal has shape `[W]` and a
2-D signal has shape `[R, C]`, and their simulation output (including CSV column names and
values) SHALL be byte-identical to the pre-change behavior.

#### Scenario: 1-D output unchanged
- **WHEN** an existing width-`W` vector signal is simulated and scoped
- **THEN** the columns are named `<id>[1]…<id>[W]` exactly as before

#### Scenario: 2-D output unchanged
- **WHEN** an existing `R×C` 2-D signal is simulated and scoped
- **THEN** the columns are named `<id>[r,c]` exactly as before

### Requirement: N-D scope rendering

A scope on a rank-N signal SHALL render one CSV column per element named
`<id>[i1,…,iN]` (1-based, row-major), generalizing the 2-D `[r,c]` form.

#### Scenario: Rank-3 scope columns
- **WHEN** a `[2, 2, 2]` signal is scoped
- **THEN** the columns are `<id>[1,1,1], <id>[1,1,2], <id>[1,2,1], …, <id>[2,2,2]`

### Requirement: N-D reshape and shape inference

`signal_reshape` SHALL accept a `shape` parameter of 1 to 6 comma-separated dimensions and
produce a signal of that shape when the element count matches its input. The shape-inference
pass SHALL propagate the full `OutShape` through shape-preserving blocks.

#### Scenario: Reshape a frame to rank-3
- **WHEN** a width-24 frame is reshaped with `shape = "2,3,4"`
- **THEN** the output is a `[2, 3, 4]` signal carrying the same 24 elements row-major

#### Scenario: Reshape element-count mismatch rejected
- **WHEN** a width-24 frame is reshaped with `shape = "2,3,5"` (30 ≠ 24)
- **THEN** lowering reports a sourced error

### Requirement: Color image as a rank-3 signal

`signal_image_source` SHALL accept a `channels` parameter so a color image flows as a
`[rows, cols, channels]` signal (interleaved), with grayscale being the `channels = 1`
special case.

#### Scenario: RGB image source shape
- **WHEN** `signal_image_source` is configured `rows=2, cols=2, channels=3`
- **THEN** it outputs a width-12 signal with shape `[2, 2, 3]`

### Requirement: Cross-stack N-D conformance

A rank-up-to-6 mflowLink signal SHALL be supported by the emit lowering, the MATLAB
Function block evaluator, and the DAP debugger inspection, reusing the existing rank-N
runtime descriptor and MLIR tensor types.

#### Scenario: 6-D signal survives the pipeline
- **WHEN** a rank-6 signal is simulated, lowered via `-emit-*`, and inspected
- **THEN** the run succeeds and the shape is preserved end-to-end (no rank truncation below 6)
