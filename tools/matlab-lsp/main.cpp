// matlab-lsp: a minimum-viable Language Server for MATLAB.
//
// Speaks the Language Server Protocol (JSON-RPC 2.0 with
// Content-Length framing over stdio) and reuses the existing
// matlabc front-end stack (Lexer -> Parser -> Resolver -> TypeInference)
// to answer editor queries.
//
// Ships these features:
//   initialize / initialized / shutdown / exit
//   textDocument/didOpen, didChange, didClose
//   textDocument/publishDiagnostics (server -> client)
//   textDocument/definition   — goto-def for user functions
//   textDocument/documentSymbol — outline view
//
// Deliberately out of scope (each a separate follow-up):
//   completion, hover, rename, workspace-wide symbols, semantic
//   highlighting, incremental parsing. See docs/lsp.md for the
//   roadmap and protocol surface we don't implement.

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"
#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"
#include "matlab/Sema/TypeInference.h"

#include "llvm/Support/JSON.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace matlab;
using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

namespace {

//===----------------------------------------------------------------------===//
// Protocol framing (Content-Length: N\r\n\r\n<body>)
//===----------------------------------------------------------------------===//

/* Read one LSP frame from stdin. Returns the body as a string on
 * success, std::nullopt on EOF or malformed input. Headers other
 * than Content-Length are parsed but ignored (the spec permits
 * additional X-* headers). */
std::optional<std::string> readFrame() {
  size_t ContentLength = 0;
  std::string Line;
  /* Header section, CRLF-terminated lines, blank line ends it. */
  while (true) {
    Line.clear();
    int c;
    while ((c = std::cin.get()) != EOF) {
      if (c == '\r') {
        if (std::cin.peek() == '\n') std::cin.get();
        break;
      }
      if (c == '\n') break;
      Line.push_back((char)c);
    }
    if (c == EOF) return std::nullopt;
    if (Line.empty()) break;
    const char Key[] = "Content-Length:";
    if (Line.compare(0, sizeof Key - 1, Key) == 0) {
      const char *s = Line.c_str() + sizeof Key - 1;
      while (*s == ' ' || *s == '\t') ++s;
      ContentLength = (size_t)std::strtoul(s, nullptr, 10);
    }
  }
  if (ContentLength == 0) return std::string{};
  std::string Body(ContentLength, '\0');
  std::cin.read(&Body[0], (std::streamsize)ContentLength);
  if (std::cin.gcount() != (std::streamsize)ContentLength) return std::nullopt;
  return Body;
}

/* Write one LSP frame to stdout. Length is computed from the
 * serialised JSON payload; stdout is flushed so the client sees
 * the message immediately. */
void writeFrame(const Value &V) {
  std::string Body;
  llvm::raw_string_ostream OS(Body);
  OS << V;
  OS.flush();
  std::cout << "Content-Length: " << Body.size() << "\r\n\r\n" << Body;
  std::cout.flush();
}

//===----------------------------------------------------------------------===//
// Per-document state
//===----------------------------------------------------------------------===//

/* Each open file owns its full parse + Sema state. We keep these
 * alive between requests so goto-definition / documentSymbol don't
 * have to re-parse the whole file for every editor query. State is
 * rebuilt end-to-end on every didChange; incremental parsing is
 * deliberately not implemented in this skeleton. */
struct Document {
  std::string URI;
  std::string Content;
  std::unique_ptr<SourceManager> SM;
  std::unique_ptr<DiagnosticEngine> Diag;
  std::unique_ptr<ASTContext> Ctx;
  std::unique_ptr<SemaContext> Sema;
  std::unique_ptr<TypeContext> TC;
  TranslationUnit *TU = nullptr;
  FileID File = 0;
};

std::unordered_map<std::string, std::unique_ptr<Document>> Docs;

/* Strip a "file://" prefix from a URI and percent-decode any %XX
 * escapes so the name we hand to SourceManager matches what the
 * client expects to see back in diagnostics. A richer URI parser
 * isn't necessary for our use. */
std::string uriToFsPath(llvm::StringRef URI) {
  llvm::StringRef S = URI;
  if (S.starts_with("file://")) S = S.drop_front(7);
  std::string Out;
  Out.reserve(S.size());
  for (size_t i = 0; i < S.size(); ++i) {
    if (S[i] == '%' && i + 2 < S.size() &&
        std::isxdigit((unsigned char)S[i + 1]) &&
        std::isxdigit((unsigned char)S[i + 2])) {
      char hex[3] = {S[i + 1], S[i + 2], 0};
      Out.push_back((char)std::strtoul(hex, nullptr, 16));
      i += 2;
    } else {
      Out.push_back(S[i]);
    }
  }
  return Out;
}

/* Rebuild the full parse + Sema pipeline for a document. Diagnostics
 * accumulate inside Diag; callers publish them after this returns. */
void reparse(Document &D) {
  D.SM = std::make_unique<SourceManager>();
  D.Diag = std::make_unique<DiagnosticEngine>(*D.SM);
  D.Ctx = std::make_unique<ASTContext>();
  D.Sema = std::make_unique<SemaContext>();
  D.TC = std::make_unique<TypeContext>();
  D.File = D.SM->addBuffer(uriToFsPath(D.URI), D.Content);

  Lexer Lx(*D.SM, D.File, *D.Diag);
  auto Toks = Lx.tokenize();
  Parser P(std::move(Toks), *D.Ctx, *D.Diag);
  D.TU = P.parseFile();
  if (D.TU) {
    Resolver R(*D.Sema, *D.TC, *D.Diag);
    R.setReplMode(false);
    R.resolve(*D.TU);
    TypeInference Inf(*D.Sema, *D.TC, *D.Diag);
    Inf.run(*D.TU);
  }
}

//===----------------------------------------------------------------------===//
// Location / position helpers (LSP uses 0-based, we use 1-based)
//===----------------------------------------------------------------------===//

/* A single {line, character} position in LSP's 0-based numbering. */
Object posObj(uint32_t Line1Based, uint32_t Col1Based) {
  return Object{
    {"line", (int64_t)(Line1Based > 0 ? Line1Based - 1 : 0)},
    {"character", (int64_t)(Col1Based > 0 ? Col1Based - 1 : 0)},
  };
}

Object rangeObj(const SourceManager &SM, SourceRange R) {
  auto B = SM.getLineColumn(R.Begin);
  /* When End isn't set (or is invalid), use Begin — an empty range
   * is still a valid LSP location. */
  SourceLocation EndLoc = R.End.isValid() ? R.End : R.Begin;
  auto E = SM.getLineColumn(EndLoc);
  /* A one-character range is common (a single identifier); keep the
   * end at least one char past the begin so editors highlight it. */
  if (B.Line == E.Line && B.Column == E.Column) E.Column += 1;
  return Object{
    {"start", posObj(B.Line, B.Column)},
    {"end", posObj(E.Line, E.Column)},
  };
}

Value locationOf(const Document &D, SourceRange R) {
  return Object{
    {"uri", D.URI},
    {"range", rangeObj(*D.SM, R)},
  };
}

//===----------------------------------------------------------------------===//
// LSP requests
//===----------------------------------------------------------------------===//

Value serverCapabilities() {
  return Object{
    /* Full document sync: we re-parse the whole buffer on didChange. */
    {"textDocumentSync", 1 /* Full */},
    {"definitionProvider", true},
    {"documentSymbolProvider", true},
  };
}

Value handleInitialize(const Object &) {
  return Object{
    {"capabilities", serverCapabilities()},
    {"serverInfo", Object{{"name", "matlab-lsp"}, {"version", "0.1"}}},
  };
}

void publishDiagnostics(const Document &D) {
  Array Arr;
  if (D.Diag) {
    for (const auto &Dg : D.Diag->diagnostics()) {
      const char *Severity = "1"; /* Error by LSP convention */
      int LSPSev = 1;
      switch (Dg.Level) {
      case DiagLevel::Error:   LSPSev = 1; break;
      case DiagLevel::Warning: LSPSev = 2; break;
      case DiagLevel::Note:    LSPSev = 3; break;
      }
      (void)Severity;
      SourceRange R = Dg.Range.Begin.isValid()
          ? Dg.Range
          : SourceRange{Dg.Loc, Dg.Loc};
      Arr.push_back(Object{
        {"range", rangeObj(*D.SM, R)},
        {"severity", LSPSev},
        {"source", "matlab-lsp"},
        {"message", Dg.Message},
      });
    }
  }
  writeFrame(Object{
    {"jsonrpc", "2.0"},
    {"method", "textDocument/publishDiagnostics"},
    {"params", Object{
      {"uri", D.URI},
      {"diagnostics", std::move(Arr)},
    }},
  });
}

/* Find the smallest NameExpr whose source range contains (Line,
 * Col). Walks the whole TU. `Line` and `Col` are 1-based. */
const NameExpr *findNameAt(const TranslationUnit &TU,
                            uint32_t Line, uint32_t Col,
                            const SourceManager &SM) {
  const NameExpr *Best = nullptr;
  uint32_t BestSpan = ~0u;
  auto spanAndContains = [&](SourceRange R) -> std::optional<uint32_t> {
    if (!R.Begin.isValid()) return std::nullopt;
    auto B = SM.getLineColumn(R.Begin);
    SourceLocation EndLoc = R.End.isValid() ? R.End : R.Begin;
    auto E = SM.getLineColumn(EndLoc);
    if (Line < B.Line || Line > E.Line) return std::nullopt;
    if (Line == B.Line && Col < B.Column) return std::nullopt;
    if (Line == E.Line && Col > E.Column) return std::nullopt;
    /* Prefer smaller enclosing span — picks the deepest NameExpr. */
    return (uint32_t)((E.Line - B.Line) * 1000 + (E.Column > B.Column ?
        E.Column - B.Column : 1));
  };
  std::function<void(const Node *)> walk = [&](const Node *N) {
    if (!N) return;
    if (auto *NE = dynamic_cast<const NameExpr *>(N)) {
      if (auto S = spanAndContains(NE->Range)) {
        if (*S < BestSpan) { BestSpan = *S; Best = NE; }
      }
    }
    /* Recurse through children — we do a brute-force walk using
     * dynamic_cast because the AST doesn't expose a generic
     * visitor and we only need it in this one place. */
    if (auto *E = dynamic_cast<const ExprStmt *>(N)) walk(E->E);
    else if (auto *A = dynamic_cast<const AssignStmt *>(N)) {
      for (Expr *L : A->LHS) walk(L);
      walk(A->RHS);
    } else if (auto *I = dynamic_cast<const IfStmt *>(N)) {
      walk(I->Cond); walk(I->Then);
      for (const auto &EI : I->Elseifs) { walk(EI.Cond); walk(EI.Body); }
      walk(I->Else);
    } else if (auto *F = dynamic_cast<const ForStmt *>(N)) {
      walk(F->Iter); walk(F->Body);
    } else if (auto *W = dynamic_cast<const WhileStmt *>(N)) {
      walk(W->Cond); walk(W->Body);
    } else if (auto *Sw = dynamic_cast<const SwitchStmt *>(N)) {
      walk(Sw->Discriminant);
      for (const auto &C : Sw->Cases) { walk(C.Value); walk(C.Body); }
    } else if (auto *T = dynamic_cast<const TryStmt *>(N)) {
      walk(T->TryBody); walk(T->CatchBody);
    } else if (auto *BL = dynamic_cast<const Block *>(N)) {
      for (const Stmt *S : BL->Stmts) walk(S);
    } else if (auto *Bi = dynamic_cast<const BinaryOpExpr *>(N)) {
      walk(Bi->LHS); walk(Bi->RHS);
    } else if (auto *U = dynamic_cast<const UnaryOpExpr *>(N)) {
      walk(U->Operand);
    } else if (auto *P = dynamic_cast<const PostfixOpExpr *>(N)) {
      walk(P->Operand);
    } else if (auto *Rn = dynamic_cast<const RangeExpr *>(N)) {
      walk(Rn->Start); walk(Rn->Step); walk(Rn->End);
    } else if (auto *C = dynamic_cast<const CallOrIndex *>(N)) {
      walk(C->Callee);
      for (Expr *Aa : C->Args) walk(Aa);
    } else if (auto *C = dynamic_cast<const CellIndex *>(N)) {
      walk(C->Callee);
      for (Expr *Aa : C->Args) walk(Aa);
    } else if (auto *FA = dynamic_cast<const FieldAccess *>(N)) {
      walk(FA->Base);
    } else if (auto *DF = dynamic_cast<const DynamicField *>(N)) {
      walk(DF->Base); walk(DF->Name);
    } else if (auto *ML = dynamic_cast<const MatrixLiteral *>(N)) {
      for (const auto &R : ML->Rows) for (Expr *C : R) walk(C);
    } else if (auto *CL = dynamic_cast<const CellLiteral *>(N)) {
      for (const auto &R : CL->Rows) for (Expr *C : R) walk(C);
    } else if (auto *AF = dynamic_cast<const AnonFunction *>(N)) {
      walk(AF->Body);
    } else if (auto *Fn = dynamic_cast<const Function *>(N)) {
      walk(Fn->Body);
      for (const Function *Nf : Fn->Nested) walk(Nf);
    } else if (auto *Sc = dynamic_cast<const Script *>(N)) {
      walk(Sc->Body);
    } else if (auto *Cd = dynamic_cast<const ClassDef *>(N)) {
      for (const Function *M : Cd->Methods) walk(M);
      for (const Function *M : Cd->StaticMethods) walk(M);
    }
  };
  if (TU.ScriptNode) walk(TU.ScriptNode);
  for (const Function *F : TU.Functions) walk(F);
  for (const ClassDef *C : TU.Classes) walk(C);
  return Best;
}

Value handleDefinition(const Object &Params) {
  const Object *TD = Params.getObject("textDocument");
  const Object *Pos = Params.getObject("position");
  if (!TD || !Pos) return nullptr;
  auto URIOpt = TD->getString("uri");
  auto LineOpt = Pos->getInteger("line");
  auto ColOpt = Pos->getInteger("character");
  if (!URIOpt || !LineOpt || !ColOpt) return nullptr;
  auto It = Docs.find(URIOpt->str());
  if (It == Docs.end() || !It->second->TU) return nullptr;
  Document &D = *It->second;
  uint32_t Line = (uint32_t)(*LineOpt) + 1; /* LSP 0-based -> 1-based */
  uint32_t Col = (uint32_t)(*ColOpt) + 1;
  const NameExpr *NE = findNameAt(*D.TU, Line, Col, *D.SM);
  if (!NE || !NE->Ref) return nullptr;
  Binding *B = NE->Ref;
  /* User function -> jump to the function's declaration. */
  if (B->Kind == BindingKind::Function && B->FuncDef) {
    return locationOf(D, B->FuncDef->Range);
  }
  /* User class -> jump to the classdef. */
  if (B->Kind == BindingKind::Class && B->ClassDef) {
    return locationOf(D, B->ClassDef->Range);
  }
  /* Variable / parameter / output -> the first use / declaration we
   * have. FirstUse is populated by the resolver. */
  if (B->FirstUse.isValid()) {
    SourceRange R{B->FirstUse, B->FirstUse};
    return locationOf(D, R);
  }
  return nullptr;
}

/* SymbolKind constants from the LSP spec. */
namespace SymKind {
  constexpr int File = 1, Module = 2, Namespace = 3, Package = 4,
                Class = 5, Method = 6, Property = 7, Field = 8,
                Constructor = 9, Enum = 10, Interface = 11, Function = 12,
                Variable = 13, Constant = 14, String = 15, Number = 16,
                Boolean = 17, Array = 18, EnumMember = 22;
}

Object symbolObj(const SourceManager &SM, llvm::StringRef Name,
                 int Kind, SourceRange Range) {
  Object R;
  R["name"] = Name;
  R["kind"] = Kind;
  R["range"] = rangeObj(SM, Range);
  R["selectionRange"] = rangeObj(SM, Range);
  return R;
}

Value handleDocumentSymbol(const Object &Params) {
  const Object *TD = Params.getObject("textDocument");
  if (!TD) return Array{};
  auto URIOpt = TD->getString("uri");
  if (!URIOpt) return Array{};
  auto It = Docs.find(URIOpt->str());
  if (It == Docs.end() || !It->second->TU) return Array{};
  Document &D = *It->second;
  Array Out;
  auto emitFn = [&](const Function &F, int Kind) -> Object {
    Object Sym = symbolObj(*D.SM, F.Name, Kind, F.Range);
    Array Children;
    /* Nested functions become children so the outline shows
     * hierarchy instead of flattening. */
    for (const Function *N : F.Nested) {
      if (N) Children.push_back(symbolObj(*D.SM, N->Name,
                                           SymKind::Function, N->Range));
    }
    if (!Children.empty()) Sym["children"] = std::move(Children);
    return Sym;
  };
  for (const Function *F : D.TU->Functions) {
    if (F) Out.push_back(emitFn(*F, SymKind::Function));
  }
  for (const ClassDef *C : D.TU->Classes) {
    if (!C) continue;
    Object Sym = symbolObj(*D.SM, C->Name, SymKind::Class, C->Range);
    Array Children;
    for (const auto &P : C->Props)
      Children.push_back(symbolObj(*D.SM, P.Name, SymKind::Property, P.Range));
    for (const Function *M : C->Methods) {
      if (!M) continue;
      int Kind = (M->Name == C->Name) ? SymKind::Constructor : SymKind::Method;
      Children.push_back(emitFn(*M, Kind));
    }
    for (const Function *M : C->StaticMethods) {
      if (M) Children.push_back(emitFn(*M, SymKind::Method));
    }
    for (auto Em : C->EnumMembers) {
      Children.push_back(symbolObj(*D.SM, Em, SymKind::EnumMember, C->Range));
    }
    (void)SymKind::File;
    if (!Children.empty()) Sym["children"] = std::move(Children);
    Out.push_back(std::move(Sym));
  }
  return Value(std::move(Out));
}

//===----------------------------------------------------------------------===//
// didOpen / didChange / didClose
//===----------------------------------------------------------------------===//

void handleDidOpen(const Object &Params) {
  const Object *TD = Params.getObject("textDocument");
  if (!TD) return;
  auto URI = TD->getString("uri");
  auto Text = TD->getString("text");
  if (!URI || !Text) return;
  auto D = std::make_unique<Document>();
  D->URI = URI->str();
  D->Content = Text->str();
  reparse(*D);
  publishDiagnostics(*D);
  Docs[D->URI] = std::move(D);
}

void handleDidChange(const Object &Params) {
  const Object *TD = Params.getObject("textDocument");
  const Array *Changes = Params.getArray("contentChanges");
  if (!TD || !Changes || Changes->empty()) return;
  auto URI = TD->getString("uri");
  if (!URI) return;
  auto It = Docs.find(URI->str());
  if (It == Docs.end()) return;
  /* We advertised TextDocumentSyncKind::Full, so each change is the
   * full new buffer contents. Take the last one — clients may send
   * a batch but only the final state matters for us. */
  const Value &Last = Changes->back();
  const Object *CO = Last.getAsObject();
  if (!CO) return;
  auto Txt = CO->getString("text");
  if (!Txt) return;
  It->second->Content = Txt->str();
  reparse(*It->second);
  publishDiagnostics(*It->second);
}

void handleDidClose(const Object &Params) {
  const Object *TD = Params.getObject("textDocument");
  if (!TD) return;
  auto URI = TD->getString("uri");
  if (!URI) return;
  Docs.erase(URI->str());
}

//===----------------------------------------------------------------------===//
// Dispatcher
//===----------------------------------------------------------------------===//

void sendResponse(const Value &Id, Value Result) {
  writeFrame(Object{
    {"jsonrpc", "2.0"},
    {"id", Id},
    {"result", std::move(Result)},
  });
}

void sendError(const Value &Id, int Code, llvm::StringRef Message) {
  writeFrame(Object{
    {"jsonrpc", "2.0"},
    {"id", Id},
    {"error", Object{{"code", Code}, {"message", Message}}},
  });
}

bool Shutdown = false;

int serve() {
  while (true) {
    auto Msg = readFrame();
    if (!Msg) break;
    if (Msg->empty()) continue;

    auto Parsed = llvm::json::parse(*Msg);
    if (!Parsed) {
      llvm::consumeError(Parsed.takeError());
      continue;
    }
    const Object *Root = Parsed->getAsObject();
    if (!Root) continue;

    auto Method = Root->getString("method");
    const Value *IdPtr = Root->get("id");
    const Object *Params = Root->getObject("params");
    Object Empty;
    if (!Params) Params = &Empty;

    if (!Method) continue;

    /* Requests (have id) vs. notifications (no id). */
    if (IdPtr) {
      if (*Method == "initialize") {
        sendResponse(*IdPtr, handleInitialize(*Params));
      } else if (*Method == "shutdown") {
        Shutdown = true;
        sendResponse(*IdPtr, nullptr);
      } else if (*Method == "textDocument/definition") {
        sendResponse(*IdPtr, handleDefinition(*Params));
      } else if (*Method == "textDocument/documentSymbol") {
        sendResponse(*IdPtr, handleDocumentSymbol(*Params));
      } else {
        /* Unimplemented request: return MethodNotFound so the editor
         * knows not to wait on it. */
        sendError(*IdPtr, -32601, "method not found");
      }
    } else {
      if (*Method == "initialized") {
        /* No-op; client just telling us it's ready. */
      } else if (*Method == "exit") {
        return Shutdown ? 0 : 1;
      } else if (*Method == "textDocument/didOpen") {
        handleDidOpen(*Params);
      } else if (*Method == "textDocument/didChange") {
        handleDidChange(*Params);
      } else if (*Method == "textDocument/didClose") {
        handleDidClose(*Params);
      }
      /* Silently drop other notifications. */
    }
  }
  return 0;
}

} // namespace

int main() {
  /* Unbuffered stdin so messages don't sit in a line buffer. */
  std::ios::sync_with_stdio(false);
  return serve();
}
