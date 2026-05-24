// RewriteDspSoForSv.h — DSP System-Object → flat-fi source rewrite.
//
// In SV-emit mode, rewrite the canonical synthesizable dsp.FIRFilter
// SO + step pattern into the flat-fi shift-register + MAC equivalent
// that the existing persistent-fi → SV regfile lane handles.  See
// the implementation comment in RewriteDspSoForSv.cpp + the design
// spec in docs/dsp_so_to_sv_bridge.md.
//
// Returns the rewritten source on success, or an empty string when no
// recognised SO pattern was present in the input (the caller then keeps
// the original source).

#ifndef MATLAB_SEMA_REWRITEDSPSOFORSV_H
#define MATLAB_SEMA_REWRITEDSPSOFORSV_H

#include <string>

namespace matlab {
namespace sema {

std::string rewriteDspSoForSv(const std::string &source);

}  // namespace sema
}  // namespace matlab

#endif  // MATLAB_SEMA_REWRITEDSPSOFORSV_H
