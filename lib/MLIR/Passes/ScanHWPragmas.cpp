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
//   `input_pipeline(N)` / `output_pipeline(N)` →
//       `hdl.input_pipeline = "N"` string attr (Phase 5.2)
//   `port(<name>, <kind>, [<signed>,] <W>[, <F>])` →
//       `hdl.ports = [<DictAttr per port>, ...]` (Phase 5.6.1).
//       Supported kinds: `fi`, `int`, `uint`, `bool`. The matching
//       func arg / result is rewritten to the declared type by
//       `ApplyPortTypePragmas`, so a function-only `.m` file emits
//       SV without a separate typed driver.
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

/// Split `<a>, <b>, <c>` (top level, no nested parens supported here
/// — the pragma surface uses simple flat arg lists) into trimmed
/// pieces. Empty input → empty output.
std::vector<std::string> splitCommas(llvm::StringRef S) {
  std::vector<std::string> Out;
  size_t I = 0;
  while (I < S.size()) {
    size_t J = S.find(',', I);
    if (J == llvm::StringRef::npos) J = S.size();
    Out.push_back(stripWS(S.substr(I, J - I)).str());
    if (J == S.size()) break;
    I = J + 1;
  }
  // Drop a single trailing empty piece (for `a, b,`).
  if (!Out.empty() && Out.back().empty()) Out.pop_back();
  return Out;
}

/// Parse one `port(...)` arg list (the substring inside the
/// parens). Forms:
///   port(<name>, fi, signed|unsigned, <W>, <F>)
///   port(<name>, int, <W>)            // signed N-bit integer
///   port(<name>, uint, <W>)           // unsigned N-bit integer
///   port(<name>, bool)                // i1
/// Builds a DictionaryAttr `{name, kind, signed, width, frac}` and
/// returns true on success. Unknown kinds / malformed pieces fail.
bool parsePortPragma(llvm::StringRef Body, mlir::MLIRContext &Ctx,
                     mlir::DictionaryAttr &Out, std::string &Err) {
  auto Parts = splitCommas(Body);
  if (Parts.size() < 2) {
    Err = "port pragma needs at least `name, kind`";
    return false;
  }
  // Strip surrounding quotes from each piece.
  for (auto &P : Parts) {
    if (P.size() >= 2 &&
        ((P.front() == '\'' && P.back() == '\'') ||
         (P.front() == '"'  && P.back() == '"')))
      P = P.substr(1, P.size() - 2);
  }
  std::string Name = Parts[0];
  std::string Kind = Parts[1];
  if (Name.empty()) { Err = "port name is empty"; return false; }
  if (Kind.empty()) { Err = "port kind is empty"; return false; }

  bool Signed = true;
  int64_t W = 0, F = 0;
  if (Kind == "fi") {
    if (Parts.size() != 5) {
      Err = "port(<name>, fi, signed|unsigned, <W>, <F>) takes 5 args";
      return false;
    }
    if (Parts[2] == "signed") Signed = true;
    else if (Parts[2] == "unsigned") Signed = false;
    else { Err = "fi sign must be `signed` or `unsigned`"; return false; }
    try { W = std::stoll(Parts[3]); } catch (...) {
      Err = "fi width is not an integer"; return false;
    }
    try { F = std::stoll(Parts[4]); } catch (...) {
      Err = "fi frac is not an integer"; return false;
    }
  } else if (Kind == "int" || Kind == "uint") {
    Signed = (Kind == "int");
    if (Parts.size() != 3) {
      Err = "port(<name>, int|uint, <W>) takes 3 args";
      return false;
    }
    try { W = std::stoll(Parts[2]); } catch (...) {
      Err = "int width is not an integer"; return false;
    }
  } else if (Kind == "bool" || Kind == "i1") {
    if (Parts.size() != 2) {
      Err = "port(<name>, bool) takes 2 args";
      return false;
    }
    Signed = false;
    W = 1;
  } else {
    Err = "unknown port kind `" + Kind + "` (want fi|int|uint|bool)";
    return false;
  }
  if (W <= 0 || W > 64) {
    Err = "port width out of range (1..64)";
    return false;
  }

  llvm::SmallVector<mlir::NamedAttribute> Fields;
  Fields.push_back({mlir::StringAttr::get(&Ctx, "name"),
                    mlir::StringAttr::get(&Ctx, Name)});
  Fields.push_back({mlir::StringAttr::get(&Ctx, "kind"),
                    mlir::StringAttr::get(&Ctx, Kind)});
  Fields.push_back({mlir::StringAttr::get(&Ctx, "signed"),
                    mlir::BoolAttr::get(&Ctx, Signed)});
  Fields.push_back({mlir::StringAttr::get(&Ctx, "width"),
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&Ctx, 64), W)});
  Fields.push_back({mlir::StringAttr::get(&Ctx, "frac"),
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&Ctx, 64), F)});
  Out = mlir::DictionaryAttr::get(&Ctx, Fields);
  return true;
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
    llvm::SmallVector<mlir::Attribute> PortAttrs;
    for (uint32_t L = StartLine; L <= MaxL; ++L) {
      auto Line = SM->getLineText(FID, L);
      llvm::StringRef LR(Line.data(), Line.size());
      scanLineForPragma(
          LR,
          [&](const std::string &Dir, const std::string &Arg) {
            // Phase 5.6.1: `port(...)` is multi-arg and accumulates
            // across lines; everything else stays one-attr-per-
            // directive. The string `Arg` we got here is already the
            // text inside the outermost `()`, so split on commas.
            if (Dir == "port") {
              mlir::DictionaryAttr Entry;
              std::string Err;
              if (parsePortPragma(Arg, *F.getContext(), Entry, Err)) {
                PortAttrs.push_back(Entry);
              } else {
                mlir::emitWarning(F.getLoc())
                    << "malformed `% hdl: port(...)` pragma: " << Err;
              }
              return;
            }
            std::string AttrName = "hdl." + Dir;
            F->setAttr(AttrName,
                       mlir::StringAttr::get(F.getContext(), Arg));
          },
          [&](const std::string &Msg) {
            mlir::emitWarning(F.getLoc()) << Msg;
          });
    }
    if (!PortAttrs.empty()) {
      F->setAttr("hdl.ports",
                 mlir::ArrayAttr::get(F.getContext(), PortAttrs));
    }
  });
  return true;
}

} // namespace mlirgen
} // namespace matlab
