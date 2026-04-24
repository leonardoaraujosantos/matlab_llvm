#pragma once

#include "matlab/AST/AST.h"

#include <iosfwd>

namespace matlab {

/// Pretty-prints a TranslationUnit as canonically-formatted MATLAB
/// source. The output round-trips through the parser back to an
/// equivalent AST.
///
/// Deliberate limitations:
///   - Comments are not preserved. The lexer strips them before
///     tokens reach the parser, so they're not present in the AST.
///     Running the formatter on a file with comments will drop them.
///   - Blank lines between top-level statements are collapsed to a
///     single newline; subtle vertical spacing is lost.
///   - Numeric literals are emitted in their parsed-canonical form
///     (e.g. `3` not `3.0`, `3.14` not `3.14000000`), using the
///     original source text when available and the typed value
///     otherwise.
void formatAST(std::ostream &OS, const TranslationUnit &TU);

} // namespace matlab
