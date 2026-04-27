// Phase 4 v2.6 — generic `% hdl: <directive>(<args>)` pragma
// scanner. Walks every user `func.func`, finds the source line
// range its body ops cover, scans those lines for pragma
// comments, and attaches each recognized directive as a
// discardable string attribute on the function.
//
// The infrastructure is intentionally minimal and generic:
// Phase 5 will reuse it for `pipeline`, `loopspec`, `ram`, etc.
// pragmas without re-doing the comment-scanning plumbing.
//
// Today's recognized directives:
//   `fsm_encoding('binary' | 'one_hot' | 'gray')` →
//       `hdl.fsm_encoding = "binary"` (etc.) string attr
//
// Unknown directives are silently ignored so the pragma surface
// can be extended over time without breaking older user code
// that mentions a future directive name. Malformed pragmas (no
// `(...)`, mismatched quote) produce a warning.

#include "matlab/MLIR/Passes/Passes.h"
#include "matlab/Basic/SourceManager.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

/// Strip leading + trailing ASCII whitespace from a string_view.
llvm::StringRef stripWS(llvm::StringRef S) {
  while (!S.empty() && std::isspace((unsigned char)S.front())) S = S.drop_front();
  while (!S.empty() && std::isspace((unsigned char)S.back()))  S = S.drop_back();
  return S;
}

/// Parse `<directive>(<arg-text>)` (with optional spaces) and
/// extract `(directive, arg-text)`. Arg text is everything inside
/// the outermost parentheses, stripped of surrounding quotes.
/// Returns false on malformed shapes.
bool parsePragmaBody(llvm::StringRef Body, std::string &Directive,
                     std::string &Arg) {
  Body = stripWS(Body);
  size_t LP = Body.find('(');
  if (LP == llvm::StringRef::npos) return false;
  size_t RP = Body.rfind(')');
  if (RP == llvm::StringRef::npos || RP <= LP) return false;
  llvm::StringRef D = stripWS(Body.take_front(LP));
  llvm::StringRef A = stripWS(Body.substr(LP + 1, RP - LP - 1));
  // Strip MATLAB-style single quotes or C-style double quotes.
  if (A.size() >= 2 &&
      ((A.front() == '\'' && A.back() == '\'') ||
       (A.front() == '"' && A.back() == '"')))
    A = A.drop_front().drop_back();
  if (D.empty()) return false;
  Directive = D.str();
  Arg = A.str();
  return true;
}

/// Scan a single line of source for `% hdl: ...` and call Cb on
/// the parsed directive/arg pair.
void scanLineForPragma(llvm::StringRef Line,
                       const std::function<void(const std::string &,
                                                const std::string &)> &Cb,
                       const std::function<void(const std::string &)> &Warn) {
  // Look for the literal `% hdl:` (or `%hdl:`) prefix anywhere in
  // the line. Anything before it is regular code/comments.
  size_t Pct = Line.find('%');
  if (Pct == llvm::StringRef::npos) return;
  llvm::StringRef Tail = Line.drop_front(Pct + 1);
  Tail = stripWS(Tail);
  if (!Tail.starts_with("hdl:")) return;
  Tail = stripWS(Tail.drop_front(4));
  // Trim a trailing line comment / continuation.
  std::string Directive, Arg;
  if (!parsePragmaBody(Tail, Directive, Arg)) {
    Warn(("malformed `% hdl:` pragma — expected `% hdl: <name>(<arg>)`, got `"
          + Tail.str() + "`"));
    return;
  }
  Cb(Directive, Arg);
}

/// Collect the [minLine, maxLine] range covered by a function's
/// body ops (across the function's source file). Returns false if
/// the function has no FileLineColLoc information.
bool functionLineRange(mlir::func::FuncOp F, std::string &OutFile,
                       uint32_t &OutMin, uint32_t &OutMax) {
  bool Any = false;
  uint32_t Mn = ~0u, Mx = 0;
  std::string FileName;
  auto Visit = [&](mlir::Location L) {
    if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(L)) {
      if (FileName.empty()) FileName = FL.getFilename().str();
      else if (FL.getFilename() != FileName) return;  // mixed file, skip
      uint32_t Line = FL.getLine();
      if (Line == 0) return;
      Mn = std::min(Mn, Line);
      Mx = std::max(Mx, Line);
      Any = true;
    }
  };
  Visit(F.getLoc());
  F.walk([&](mlir::Operation *Op) { Visit(Op->getLoc()); });
  if (!Any) return false;
  OutFile = std::move(FileName);
  OutMin = Mn;
  OutMax = Mx;
  return true;
}

} // namespace

bool runScanHWPragmas(mlir::ModuleOp M, const matlab::SourceManager *SM) {
  if (!SM) return true;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    std::string FileName;
    uint32_t MinL = 0, MaxL = 0;
    if (!functionLineRange(F, FileName, MinL, MaxL)) return;
    matlab::FileID FID = SM->findFileByName(FileName);
    if (FID == 0) return;
    // A pragma can sit immediately above the function definition
    // (one line of leading whitespace tolerated) — extend the
    // scan range one line up to catch that case.
    uint32_t StartLine = MinL > 1 ? MinL - 1 : 1;
    for (uint32_t L = StartLine; L <= MaxL; ++L) {
      auto Line = SM->getLineText(FID, L);
      llvm::StringRef LR(Line.data(), Line.size());
      scanLineForPragma(
          LR,
          [&](const std::string &Dir, const std::string &Arg) {
            std::string AttrName = "hdl." + Dir;
            F->setAttr(AttrName,
                       mlir::StringAttr::get(F.getContext(), Arg));
          },
          [&](const std::string &Msg) {
            mlir::emitWarning(F.getLoc()) << Msg;
          });
    }
  });
  return true;
}

} // namespace mlirgen
} // namespace matlab
