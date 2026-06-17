# MATLAB Source Formatter Spec

## Purpose
Document the observed behavior of the `.m` source formatter exposed as `matlabc -format`. It pretty-prints a parsed `TranslationUnit` as canonically-formatted MATLAB source via `formatAST`, and is reused by the `.mflow` emitter (`formatExpr`) to render canonical expression text. This spec records the canonical formatting rules and their deliberate limitations as they exist today (src: include/matlab/AST/Formatter.h, lib/AST/Formatter.cpp, tools/matlabc/main.cpp).

## Requirements

### Requirement: Format mode emits canonical source
The system SHALL, when invoked with `-format`, parse the input `.m` file and print the canonically-formatted MATLAB source of its `TranslationUnit` to stdout.

#### Scenario: File formatted to stdout
- **WHEN** the user runs `matlabc -format foo.m`
- **THEN** the system SHALL print the canonically-formatted source via `formatAST` and exit non-zero only if diagnostics report errors (src: tools/matlabc/main.cpp `Mode::Format` -> `formatAST(std::cout, *TU)`)

#### Scenario: Format and emit-matlab share the path
- **WHEN** the user runs `matlabc -emit-matlab foo.m`
- **THEN** the system SHALL produce the same canonical formatting as `-format`, since both modes route through `formatAST` (src: tools/matlabc/main.cpp `Mode::Format || Mode::EmitMatlab`)

### Requirement: Parser round-trip
The system SHALL emit source that round-trips through the parser back to an equivalent AST.

#### Scenario: Formatted output re-parses
- **WHEN** formatter output is fed back into the parser
- **THEN** the system SHALL produce an AST equivalent to the one that generated the output (src: include/matlab/AST/Formatter.h "The output round-trips through the parser back to an equivalent AST")

### Requirement: Canonical numeric literal form
The system SHALL emit numeric literals in their parsed-canonical form, preferring the original source text when available and the typed value otherwise.

#### Scenario: Literals not zero-padded
- **WHEN** the source contains `3` and `3.14`
- **THEN** the system SHALL emit `3` (not `3.0`) and `3.14` (not `3.14000000`) (src: include/matlab/AST/Formatter.h numeric-literal note)

### Requirement: Comments are not preserved
The system SHALL drop comments when formatting, because the lexer strips comments before tokens reach the parser so they are absent from the AST.

#### Scenario: Comment-bearing file loses comments
- **WHEN** the user formats a file containing `% comment` lines
- **THEN** the system SHALL emit output with those comments removed (src: include/matlab/AST/Formatter.h "Comments are not preserved")

### Requirement: Blank-line collapsing
The system SHALL collapse blank lines between top-level statements to a single newline, not preserving subtle vertical spacing.

#### Scenario: Multiple blank lines collapsed
- **WHEN** the source has several blank lines between two top-level statements
- **THEN** the system SHALL emit a single newline between them (src: include/matlab/AST/Formatter.h "Blank lines ... are collapsed to a single newline")

### Requirement: Single-expression formatting helper
The system SHALL provide `formatExpr` to pretty-print a single expression with no trailing semicolon or newline, used by the `.mflow` emitter to fill `data.cond` / `data.iter` / `data.expression` / `data.rhs` / `data.value` fields.

#### Scenario: Expression rendered for a flowchart field
- **WHEN** the `.mflow` emitter needs the canonical text of a condition expression
- **THEN** the system SHALL render it via `formatExpr` with no trailing semicolon or newline (src: include/matlab/AST/Formatter.h `formatExpr`; lib/Flowchart/ASTToGraph.cpp)
