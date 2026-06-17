# Flowchart Frontend Spec

## Purpose
The flowchart frontend consumes `.mflow` JSON graph programs (produced by the MatForge IDE) and lowers them into the same `TranslationUnit` AST as the textual `.m` frontend, so every existing Sema pass and `-emit-*` backend works unchanged. It also provides the inverse `-emit-mflow` emitter, an mflowlink block model with an in-process simulator, and a subsystem-to-MATLAB lowering. (src: lib/Flowchart/Loader.cpp, lib/Flowchart/GraphToAST.cpp, lib/Flowchart/ASTToGraph.cpp, doc: docs/flowchart_frontend.md)

## Requirements

### Requirement: Loading and validating `.mflow` documents
The system SHALL parse `.mflow` JSON into a typed `FlowDoc` (flows, nodes, edges, settings) using a hand-rolled reader that records per-field byte offsets, and SHALL validate the document before any AST construction. (src: lib/Flowchart/Loader.cpp, src: include/matlab/Flowchart/Loader.h, doc: docs/flowchart_frontend.md)

#### Scenario: Structural validation
- **WHEN** a program flow is loaded
- **THEN** the system SHALL require unique node ids per flow, resolve every edge endpoint against the declared node ports, and require exactly one `start` node in a control-flow `program` flow, reporting failures through the diagnostic engine (src: lib/Flowchart/Loader.cpp, test/Flowchart/Errors/duplicate_node.mflow, test/Flowchart/Errors/no_start.mflow, test/Flowchart/Errors/two_starts.mflow)

#### Scenario: Bad edge endpoint
- **WHEN** an edge references a non-existent node or a port not declared on that node
- **THEN** the system SHALL emit a sourced diagnostic identifying the offending node/port (src: lib/Flowchart/Loader.cpp, test/Flowchart/Errors/bad_edge_node.mflow, test/Flowchart/Errors/bad_edge_port.mflow)

#### Scenario: Unreachable nodes warn but do not block
- **WHEN** a node is not reachable from `start` (e.g. disconnected palette nodes)
- **THEN** the system SHALL emit a warning and skip the node rather than failing the load (src: lib/Flowchart/Loader.cpp, test/Flowchart/disconnected_palette.mflow)

### Requirement: Graph-to-AST reduction
The system SHALL reduce a validated control-flow graph into a structured `TranslationUnit` AST, mapping linear node chains to statement blocks and reconverging branches into structured control flow. (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Linear chain
- **WHEN** a flow is a linear `start → ... → end` chain of statement blocks (variable, expression, display, assignment, function_call, constant, matrix_literal)
- **THEN** the system SHALL emit the corresponding statements in order, reusing the textual `Lexer`+`Parser` for string-valued `data` fields (src: lib/Flowchart/GraphToAST.cpp, test/Flowchart/linear.mflow)

#### Scenario: Branch reconvergence
- **WHEN** an `if` node's `true`/`false` branches reconverge at a common point
- **THEN** the system SHALL locate the join via a two-pointer `findJoin` walk and emit an `IfStmt` whose `Then`/`Else` are the reduced sub-regions, dropping an empty `Else` for if-without-else (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Irreducible graph
- **WHEN** branches do not reconverge (irreducible CFG)
- **THEN** the system SHALL emit a diagnostic rather than synthesizing a `goto` (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

### Requirement: Loops, terminators, switch, and try blocks
The system SHALL reduce `for`/`while` loop nodes, `break`/`continue`/`return` terminators, and `switch`/`try` block kinds into the matching AST statements. (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Loop reduction
- **WHEN** a `for` or `while` node has a `body` port that loops back and a `done` continuation port
- **THEN** the system SHALL emit a `ForStmt` or `WhileStmt` whose body is the reduced back-edge region and whose continuation follows the `done` port (src: lib/Flowchart/GraphToAST.cpp)

#### Scenario: Switch reduction
- **WHEN** a `switch` node exposes `case_0`…`case_<N-1>` and `default` ports
- **THEN** the system SHALL walk each branch independently, compute the multi-way join by iterated pairwise `findJoin`, and emit a `SwitchStmt` (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

### Requirement: Sub-flows and custom blocks
The system SHALL lower non-`program` flows into top-level `Function`s and SHALL resolve `function_definition`, `subflow_call`, and `custom` blocks into calls and function bodies. (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Sub-flow lifting
- **WHEN** a `function`-kind flow is referenced
- **THEN** the system SHALL lift it into a top-level `Function` with the flow's signature and emit `subflow_call` blocks as `lhs = name(args)` calls, rejecting duplicate function names (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Custom block provenance
- **WHEN** a `custom` block specifies its body via `data.source` (inline MATLAB), `data.path` (relative `.m` file), or `data.library_id` (block-search-path lookup)
- **THEN** the system SHALL require exactly one provenance, insert the resulting `Function` at most once per unique name, and emit a call site for each block instance (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

### Requirement: `-emit-mflow` AST-to-graph emission
The system SHALL emit a canonical `.mflow` document from a `TranslationUnit`, as the structural inverse of the reducer, with idempotent repeat emission. (src: lib/Flowchart/ASTToGraph.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Statement-to-block mapping
- **WHEN** a TU is emitted to `.mflow`
- **THEN** the system SHALL map linear statements to block kinds, `IfStmt`/`ForStmt`/`WhileStmt`/`SwitchStmt`/`TryStmt` to their control block kinds with proper ports and back-edges, and each `Function` to a `function`-kind sub-flow (src: lib/Flowchart/ASTToGraph.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Idempotent emission
- **WHEN** the same TU is emitted twice (or round-tripped `.m → .mflow → .m → .mflow`)
- **THEN** the system SHALL produce byte-identical output via stable `n_<kind>_<counter>` ids and canonical formatting, with an optional `--preserve-layout` merge of prior `ui.position` values (src: lib/Flowchart/ASTToGraph.cpp, doc: docs/flowchart_frontend.md)

### Requirement: Cross-backend and debug parity with the textual frontend
The system SHALL route `.mflow` inputs through the same Sema and `-emit-*` backends as `.m`, and SHALL support DAP/LSP on `.mflow` programs with per-block source mapping. (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

#### Scenario: Backend reuse
- **WHEN** a `.mflow` program is compiled with any `-emit-*` mode
- **THEN** the system SHALL produce output equivalent to compiling the round-tripped `-emit-matlab` source of the same program (src: lib/Flowchart/GraphToAST.cpp, test/Flowchart/CrossBackend)

#### Scenario: Block-aware debugging
- **WHEN** a debugger sets a breakpoint on a `.mflow` block
- **THEN** the system SHALL rewrite synthesized statements' source ranges to the originating block's byte offset and record a `(file_id, line) → block_id` map so breakpoints fire and stack frames surface the active block (src: lib/Flowchart/GraphToAST.cpp, doc: docs/flowchart_frontend.md)

### Requirement: mflowlink block model and simulation
The system SHALL build an mflowlink signal-flow block model from a `.mflow` document, classify block sample times, break algebraic loops, and run an in-process simulator. (src: include/matlab/Flowchart/MflowLinkModel.h, src: lib/Flowchart/MflowLinkSim.cpp, doc: docs/mflowlink_blocks.md)

#### Scenario: Sample-time classification and loop breaking
- **WHEN** a signal-flow model is built
- **THEN** the system SHALL classify each block's sample time (Continuous, Discrete, Constant, FixedInMinor), track continuous/discrete state counts, and mark loop-breaker blocks (Integrator, Unit Delay, strictly-proper transfer functions) so their edges are dropped from the execution-order topological sort (src: include/matlab/Flowchart/MflowLinkModel.h)

#### Scenario: In-process simulation
- **WHEN** an mflowlink model is simulated
- **THEN** the system SHALL step blocks in execution order with continuous-state integration and produce logged signal output (src: lib/Flowchart/MflowLinkSim.cpp, examples/mflowlink/bouncing_ball.mflow, test/Flowchart/Simulate)

### Requirement: Subsystem-to-MATLAB lowering
The system SHALL lower signal-flow subsystems into MATLAB function ASTs. (src: lib/Flowchart/SubsystemToMatlab.cpp, src: include/matlab/Flowchart/SubsystemToMatlab.h)

#### Scenario: Subsystem export
- **WHEN** a signal-flow subsystem is converted
- **THEN** the system SHALL emit a MATLAB function AST representing the subsystem's stateless block computation (src: lib/Flowchart/SubsystemToMatlab.cpp, test/Flowchart/EmitSubsystem)
