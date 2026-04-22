#pragma once

#include "matlab/MIR/MIR.h"

#include <iosfwd>

namespace matlab {
namespace mir {

// MLIR-style textual dump for a Module. Numbered SSA values (%0, %1, ...),
// op names in snake_case (matlab.* prefix), nested regions indented.
void printModule(std::ostream &OS, const Module &M);

void printOp(std::ostream &OS, const Op &O, unsigned Indent = 0);

} // namespace mir
} // namespace matlab
