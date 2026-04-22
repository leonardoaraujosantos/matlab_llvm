#pragma once

#include "matlab/AST/AST.h"

#include <iosfwd>

namespace matlab {

void dumpAST(std::ostream &OS, const Node &N);

} // namespace matlab
