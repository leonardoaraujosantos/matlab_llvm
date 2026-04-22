#pragma once

#include "matlab/AST/AST.h"

#include <iosfwd>

namespace matlab {

// Dumps the AST annotated with resolver + type-inference results.
// Each Expr prints its inferred type in brackets. Each NameExpr shows the
// resolved binding kind. Each CallOrIndex shows whether it's a Call or Index.
void dumpSema(std::ostream &OS, const Node &N);

} // namespace matlab
