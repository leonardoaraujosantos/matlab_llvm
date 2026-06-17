# Language Server (matlab-lsp) Spec

## Purpose
Document the observed behavior of `matlab-lsp`, an LSP server built on the same lex / parse / Sema stack as `matlabc`. It speaks JSON-RPC 2.0 over stdio with `Content-Length` framing and exposes diagnostics, goto-definition, and document outline to any LSP-capable editor. This spec records exactly which requests are supported today and which are deliberately unimplemented (doc: docs/lsp.md; src: tools/matlab-lsp/main.cpp).

## Requirements

### Requirement: Initialize and shutdown lifecycle
The system SHALL implement the LSP `initialize` / `initialized` handshake and the `shutdown` / `exit` lifecycle, advertising `definitionProvider` and `documentSymbolProvider` capabilities.

#### Scenario: Editor initializes the server
- **WHEN** an LSP client sends `initialize`
- **THEN** the system SHALL respond with capabilities and serverInfo, advertising `definitionProvider=true` and `documentSymbolProvider=true` (src: tools/matlab-lsp/main.cpp `definitionProvider` / `documentSymbolProvider`)

#### Scenario: Exit without prior shutdown
- **WHEN** the client sends `exit` without a preceding `shutdown`
- **THEN** the system SHALL exit with code 1 (doc: docs/lsp.md "Protocol cheat sheet")

### Requirement: Full-document synchronization
The system SHALL track open documents and re-parse the entire buffer on every change (`textDocumentSync = Full`), handling `textDocument/didOpen`, `didChange`, and `didClose`.

#### Scenario: Document state dropped on close
- **WHEN** the client sends `textDocument/didClose`
- **THEN** the system SHALL drop the document's tracked state (doc: docs/lsp.md "Protocol cheat sheet")

### Requirement: Diagnostics on open and change
The system SHALL re-parse on `didOpen` and `didChange` and publish the resulting diagnostics to the client via `textDocument/publishDiagnostics`.

#### Scenario: Diagnostics republished after an edit
- **WHEN** the user edits an open `.m` buffer
- **THEN** the system SHALL re-parse the full buffer and send a fresh `textDocument/publishDiagnostics` notification (src: tools/matlab-lsp/main.cpp `publishDiagnostics`)

### Requirement: Goto-definition
The system SHALL answer `textDocument/definition` for user functions, user classes, and variables, returning a `Location` or `null`.

#### Scenario: Jump to a user function definition
- **WHEN** the client sends `textDocument/definition` over a call to a user-defined function
- **THEN** the system SHALL return the `Location` of that function's definition, or `null` when no definition resolves (doc: docs/lsp.md "Shipped features")

### Requirement: Document symbol outline
The system SHALL answer `textDocument/documentSymbol` with the file's functions, classes, properties, and methods for the editor's outline view.

#### Scenario: Outline of a classdef file
- **WHEN** the client requests `textDocument/documentSymbol` for a file containing a class
- **THEN** the system SHALL return a `DocumentSymbol[]` listing the class plus its properties and methods (doc: docs/lsp.md "Shipped features")

### Requirement: Unknown-method handling
The system SHALL respond to unknown requests with the JSON-RPC error `-32601 MethodNotFound` and silently drop unknown notifications.

#### Scenario: Hover and completion are not implemented
- **WHEN** the client sends `textDocument/hover` or `textDocument/completion`
- **THEN** the system SHALL return `-32601 MethodNotFound` because these requests are deliberately not implemented today (doc: docs/lsp.md "Not implemented (and why)"; src: tools/matlab-lsp/main.cpp method dispatch)
