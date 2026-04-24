# Language Server (`matlab-lsp`)

A small LSP server built on the same lex / parse / Sema stack
matlabc uses. Speaks JSON-RPC 2.0 over stdio with `Content-Length`
framing; drops into any editor that can launch an LSP server.

## Shipped features

| Feature                                                        | Method                             |
| -------------------------------------------------------------- | ---------------------------------- |
| Initialize handshake                                           | `initialize`, `initialized`        |
| Clean shutdown                                                  | `shutdown`, `exit`                 |
| Open / modify / close documents                                 | `textDocument/didOpen` / `didChange` / `didClose` |
| Diagnostics on open + every change                              | `textDocument/publishDiagnostics`  |
| Goto-definition (user functions, user classes, variables)       | `textDocument/definition`          |
| Outline view (functions, classes, properties, methods)          | `textDocument/documentSymbol`      |

The server re-parses the full buffer on every change
(`textDocumentSync = Full`). Incremental parsing is deliberately
not implemented — re-parsing a typical `.m` file takes well under a
millisecond on our stack, so the complexity isn't worth it yet.

## Editor setup

The binary is at `build/matlab-lsp` after `just build`. The editor
just needs to know: "when you see a `.m` file, launch this binary
and speak LSP to it."

### Neovim (nvim-lspconfig-free example)

```lua
vim.api.nvim_create_autocmd("FileType", {
  pattern = "matlab",
  callback = function()
    vim.lsp.start({
      name = "matlab-lsp",
      cmd = { "/path/to/matlab_llvm/build/matlab-lsp" },
      root_dir = vim.fs.dirname(vim.fs.find({".git"}, { upward = true })[1]),
    })
  end,
})
```

### VS Code (`settings.json` via a generic LSP extension)

Any "generic LSP" extension (there are several, e.g.
`ms-vscode.language-server`) will accept:

```json
{
  "matlab-lsp": {
    "command": "/path/to/matlab_llvm/build/matlab-lsp",
    "filetypes": ["matlab"]
  }
}
```

### Helix

```toml
[language-server.matlab-lsp]
command = "/path/to/matlab_llvm/build/matlab-lsp"

[[language]]
name = "matlab"
file-types = ["m"]
language-servers = ["matlab-lsp"]
```

## Not implemented (and why)

Each item below is a separate follow-up. They're left out of this
skeleton to keep the first commit reviewable.

- `textDocument/completion` — needs a ranked list of in-scope names
  under the cursor, plus a snippet model for function calls.
- `textDocument/hover` — would surface the Sema-inferred type of
  the expression under the cursor; easy to start, but getting the
  type text readable (matrix shapes, function signatures, class
  properties) takes real work.
- `textDocument/rename` — needs an AST-driven refactoring pass that
  updates every use of a binding in the same file.
- `workspace/symbol` — workspace-wide symbol search; requires
  cross-file resolution, which Sema doesn't do today (each file is
  its own TU).
- `textDocument/semanticTokens` — syntax highlighting that tracks
  Sema binding kinds (`var` vs. `function` vs. `class` vs. `builtin`).
- Incremental parsing — we reparse the whole file on every change.
  Fast enough today; revisit if responsiveness regresses on large
  files.

## Protocol cheat sheet

The skeleton's dispatch table, abbreviated:

```
request  initialize              -> capabilities + serverInfo
request  shutdown                -> null
request  textDocument/definition -> Location or null
request  textDocument/documentSymbol -> DocumentSymbol[]
notif    initialized             (no-op)
notif    exit                    -> exit(0) or exit(1) if no shutdown
notif    textDocument/didOpen    -> reparse + publishDiagnostics
notif    textDocument/didChange  -> reparse + publishDiagnostics
notif    textDocument/didClose   -> drop state
notif    textDocument/publishDiagnostics  (server -> client)
```

Unknown requests get `-32601 MethodNotFound`; unknown notifications
are silently dropped (per LSP convention).
