# Source Round-Trip Emitters Spec

## Purpose
Documents the observed behavior of `matlabc`'s reverse emitters: `-emit-matlab` (alias `-emit-m`) re-serializes the parsed AST to canonical `.m` source, and `-emit-mflow` (alias `-emit-flow`) serializes the AST to a `.mflow` IDE flowchart diagram. Both accept either `.m` or `.mflow` input, enabling round-trips between textual source and the visual diagram. Output is deterministic and canonical so that re-emission stabilizes byte-for-byte.

## Requirements

### Requirement: Emit canonical MATLAB source
The system SHALL, in `-emit-matlab` / `-emit-m` mode, format the parsed translation unit as canonical MATLAB source on stdout.

#### Scenario: Re-serializing a script
- **WHEN** a user runs `matlabc -emit-matlab foo.m`
- **THEN** the system SHALL print canonical `.m` source to stdout with normalized spacing and precedence-driven parentheses (src: lib/AST/Formatter.cpp; src: include/matlab/AST/Formatter.h)

#### Scenario: Comments are not preserved
- **WHEN** the input contains comments
- **THEN** the system SHALL omit them because the lexer strips comments before the AST is built (src: include/matlab/AST/Formatter.h)

### Requirement: Diagram-to-source conversion
The system SHALL accept a `.mflow` diagram as input to `-emit-matlab` and emit the equivalent MATLAB source.

#### Scenario: Converting a diagram to source
- **WHEN** a user runs `matlabc -emit-matlab foo.mflow`
- **THEN** the system SHALL load the diagram, build the AST, and print the equivalent canonical `.m` source (src: tools/matlabc/main.cpp; src: lib/Flowchart/ASTToGraph.cpp)

### Requirement: Emit .mflow diagram
The system SHALL, in `-emit-mflow` / `-emit-flow` mode, serialize the translation unit to a `.mflow` JSON document on stdout following the flowchart schema.

#### Scenario: Source to diagram
- **WHEN** a user runs `matlabc -emit-mflow foo.m`
- **THEN** the system SHALL print a `.mflow` JSON document with the schema/version header, a flow per program and function, and nodes/edges for each statement (src: lib/Flowchart/ASTToGraph.cpp; doc: docs/flowchart_schema.md)

### Requirement: Deterministic canonical JSON
The system SHALL emit `.mflow` JSON deterministically with stable node IDs, alphabetically ordered keys, and auto-layout positions so re-emission is byte-identical.

#### Scenario: Stable re-emission
- **WHEN** the same diagram shape is emitted twice
- **THEN** the system SHALL produce byte-identical JSON (src: lib/Flowchart/ASTToGraph.cpp; src: include/matlab/Flowchart/ASTToGraph.h)

### Requirement: Layout preservation
The system SHALL, when `--preserve-layout` references an existing `.mflow`, copy node positions for matching nodes and auto-layout the rest.

#### Scenario: Re-emitting with preserved positions
- **WHEN** `-emit-mflow input --preserve-layout ref.mflow` runs
- **THEN** the system SHALL keep `ui.position` for nodes matched by `(flow_id, node_id)` and auto-layout unmatched nodes (src: tools/matlabc/main.cpp; doc: docs/flowchart_frontend.md)

### Requirement: Round-trip idempotency testing
The system SHALL verify the emitters with golden comparisons and an idempotency round-trip that stabilizes after the first canonicalization.

#### Scenario: emit-matlab goldens
- **WHEN** the `flowchart-emit-matlab-tests` lane runs each `.mflow` fixture
- **THEN** the system SHALL diff `-emit-matlab` output against its `.expected` golden (src: test/Flowchart/EmitMatlab)

#### Scenario: emit-mflow idempotency
- **WHEN** the `flowchart-emit-mflow-tests` lane round-trips input through `-emit-mflow` and `-emit-matlab` repeatedly
- **THEN** the system SHALL produce byte-identical `.mflow` output from the second iteration onward (src: test/Flowchart/EmitMflow)
