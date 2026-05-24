// RewriteDspSoForSv.cpp — DSP System-Object → flat-fi source rewrite
// for the -emit-systemverilog pipeline (Category-1 v1 of
// `LowerDspSystemObjects`; see docs/dsp_so_to_sv_bridge.md for the full
// design + the eventual MLIR-pass path).
//
// What this does:
//   In SV-emit mode, scan the user source for the canonical
//   synthesizable dsp.FIRFilter construction + step pattern:
//
//     persistent VAR;
//     if isempty(VAR)
//         VAR = dsp.FIRFilter('Numerator', fi([COEFFS], S, W, F));
//     end
//     RESULT = VAR(fi(EXPR, S', W', F'));     % or VAR(EXPR)
//
//   …and substitute the flat-fi shift-register + MAC equivalent that the
//   existing persistent-fi → SV regfile lane handles bit-identically.
//   After the rewrite the source no longer mentions `dsp.FIRFilter`, so
//   the prelude detection skips dsp_classdefs.m and the SV pipeline
//   never sees the SO machinery (matlab_obj_new / matlab_dsp_iir_step
//   et al. — those are non-synthesizable and would fail HWLegalize).
//
// Why source-level (v1):
//   The MLIR pass form (a true `LowerDspSystemObjects` that rewrites the
//   lowered IR) needs to additionally strip the dsp_FIRFilter classdef
//   method funcs from the module — otherwise their bodies fail
//   HWLegalize even when the user function is fine.  Source-level
//   substitution sidesteps that entirely (the prelude is never loaded)
//   and ships a working bridge today.  The eventual MLIR pass remains
//   the documented follow-on for users who can't rewrite their source
//   (e.g. SO instances passed through other functions).

#include "matlab/Sema/RewriteDspSoForSv.h"

#include <cctype>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace matlab {
namespace sema {

namespace {

/// Count comma- or whitespace-separated tokens in `s` — for a
/// coefficient list like " 1, 2 3 4 3, 2, 1 " returns 7.
int countTokens(const std::string &s) {
    int n = 0;
    bool inTok = false;
    for (char c : s) {
        bool ws = std::isspace(static_cast<unsigned char>(c)) || c == ',';
        if (!ws && !inTok) { inTok = true; ++n; }
        if (ws) inTok = false;
    }
    return n;
}

/// Inside the regex captured `step input expression`, recognise either
/// `fi(EXPR, S, W, F)` (returns the inner expr + the cast type) or
/// a bare expression (returns the expression + no cast).  Returns
/// `true` if the cast form was detected.
bool parseFiCast(const std::string &inputExpr,
                 std::string &innerExpr,
                 int &sgn, int &wl, int &fl) {
    static const std::regex fiPat(
        R"(^\s*fi\s*\(\s*([^,]+?)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$)",
        std::regex::ECMAScript);
    std::smatch m;
    if (!std::regex_match(inputExpr, m, fiPat)) {
        innerExpr = inputExpr;
        return false;
    }
    innerExpr = m[1].str();
    sgn = std::stoi(m[2].str());
    wl  = std::stoi(m[3].str());
    fl  = std::stoi(m[4].str());
    return true;
}

}  // namespace

std::string rewriteDspSoForSv(const std::string &source) {
    /* Match the canonical synthesizable SO+step pattern.  The regex is
     * deliberately strict — anything that doesn't match exactly bails to
     * the empty-string return, and the standard SV pipeline fires its
     * normal "no synthesizable form" error for the SO surface, which is
     * the right failure mode for non-canonical inputs.
     *
     * Groups:
     *   1 — leading whitespace / indentation (preserved on output).
     *   2 — persistent object variable name (e.g. firFilt).
     *   3 — coefficient list contents between the brackets.
     *   4 — coefficient fi sign flag.
     *   5 — coefficient fi word length.
     *   6 — coefficient fi fraction length.
     *   7 — result variable name on the step assignment.
     *   8 — the full step-call argument expression. */
    static const std::regex pat(
        R"(([ \t]*)persistent\s+(\w+)\s*;\s*)"
        R"(if\s+isempty\s*\(\s*\2\s*\)\s*)"
        R"(\2\s*=\s*dsp\.FIRFilter\s*\(\s*'Numerator'\s*,\s*)"
        R"(fi\s*\(\s*\[\s*([0-9\s\.,+-]+?)\s*\]\s*,\s*)"
        R"((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)\s*;\s*)"
        R"(end\s*)"
        R"((\w+)\s*=\s*\2\s*\(\s*([^)]+(?:\([^)]*\)[^)]*)*)\s*\)\s*;)",
        std::regex::ECMAScript);

    std::smatch m;
    if (!std::regex_search(source, m, pat)) return std::string();

    std::string indent    = m[1].str();
    std::string objVar    = m[2].str();
    std::string coeffs    = m[3].str();
    int         coefS     = std::stoi(m[4].str());
    int         coefW     = std::stoi(m[5].str());
    int         coefF     = std::stoi(m[6].str());
    std::string resultVar = m[7].str();
    std::string stepArg   = m[8].str();

    int taps = countTokens(coeffs);
    if (taps < 2) return std::string();  // 1-tap FIR has no delay line

    /* The state / input fi-type matches the step call's fi() cast when
     * present, otherwise falls back to the coefficient type. */
    std::string innerExpr;
    int inS = coefS, inW = coefW, inF = coefF;
    parseFiCast(stepArg, innerExpr, inS, inW, inF);

    std::ostringstream out;
    out << indent << "h = fi([" << coeffs << "], "
        << coefS << ", " << coefW << ", " << coefF << ");\n";
    out << indent << "persistent delay_line;\n";
    out << indent << "if isempty(delay_line)\n";
    out << indent << "    delay_line = fi(zeros(1, " << taps << "), "
        << inS << ", " << inW << ", " << inF << ");\n";
    out << indent << "end\n";
    out << indent << "delay_line = [fi(" << innerExpr << ", "
        << inS << ", " << inW << ", " << inF
        << "), delay_line(1:" << (taps - 1) << ")];\n";
    for (int k = 1; k <= taps; ++k)
        out << indent << "p" << k
            << " = delay_line(" << k << ") * h(" << k << ");\n";
    out << indent << resultVar << " = p1";
    for (int k = 2; k <= taps; ++k) out << " + p" << k;
    out << ";\n";

    std::string rewritten = source;
    rewritten.replace(m.position(0), m.length(0), out.str());
    return rewritten;
}

}  // namespace sema
}  // namespace matlab
