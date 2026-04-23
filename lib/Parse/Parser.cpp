#include "matlab/Parse/Parser.h"

#include <cassert>
#include <string>

namespace matlab {

Parser::Parser(std::vector<Token> Tokens, ASTContext &Ctx,
               DiagnosticEngine &Diag)
    : Toks(std::move(Tokens)), Ctx(Ctx), Diag(Diag) {
  ScopeStack.emplace_back(); // global scope
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

bool Parser::isStatementTerminator() const {
  return cur().isOneOf({TokenKind::semi, TokenKind::comma,
                        TokenKind::newline, TokenKind::eof});
}

void Parser::skipStatementTerminators() {
  while (cur().isOneOf(
      {TokenKind::semi, TokenKind::comma, TokenKind::newline}))
    ++Idx;
}

bool Parser::expect(TokenKind K, const char *What) {
  if (consume(K)) return true;
  Diag.error(cur().Loc,
             std::string("expected ") + What + ", found '" +
                 std::string(cur().Text) + "'");
  return false;
}

bool Parser::atEndOfBlock() const {
  return cur().isOneOf({TokenKind::kw_end, TokenKind::kw_else,
                        TokenKind::kw_elseif, TokenKind::kw_case,
                        TokenKind::kw_otherwise, TokenKind::kw_catch,
                        TokenKind::eof});
}

// Error recovery: advance until we hit a statement terminator or block end.
void Parser::synchronize() {
  while (!cur().is(TokenKind::eof) && !isStatementTerminator() &&
         !atEndOfBlock())
    ++Idx;
  skipStatementTerminators();
}

std::string Parser::unescapeDoubleQuoted(std::string_view Raw) {
  // Raw includes the outer quotes.
  std::string Out;
  Out.reserve(Raw.size());
  for (size_t i = 1; i + 1 < Raw.size(); ++i) {
    if (Raw[i] == '"' && i + 2 < Raw.size() && Raw[i + 1] == '"') {
      Out.push_back('"');
      ++i;
    } else {
      Out.push_back(Raw[i]);
    }
  }
  return Out;
}

std::string Parser::unescapeSingleQuoted(std::string_view Raw) {
  std::string Out;
  Out.reserve(Raw.size());
  for (size_t i = 1; i + 1 < Raw.size(); ++i) {
    if (Raw[i] == '\'' && i + 2 < Raw.size() && Raw[i + 1] == '\'') {
      Out.push_back('\'');
      ++i;
    } else {
      Out.push_back(Raw[i]);
    }
  }
  return Out;
}

//===----------------------------------------------------------------------===//
// Top-level parseFile
//===----------------------------------------------------------------------===//

TranslationUnit *Parser::parseFile() {
  auto *TU = Ctx.make<TranslationUnit>();
  SourceLocation Start = cur().Loc;
  TU->Range.Begin = Start;

  skipStatementTerminators();

  if (at(TokenKind::kw_classdef)) {
    // A class file. MATLAB allows exactly one classdef per file, optionally
    // followed by local functions. We accept the same shape.
    if (auto *C = parseClassDef()) TU->Classes.push_back(C);
    skipStatementTerminators();
    while (at(TokenKind::kw_function)) {
      if (auto *F = parseFunction()) TU->Functions.push_back(F);
      skipStatementTerminators();
    }
    if (!at(TokenKind::eof))
      Diag.error(cur().Loc, "stray tokens after classdef");
  } else if (at(TokenKind::kw_function)) {
    // A .m file that is a function file. May contain multiple functions.
    while (at(TokenKind::kw_function)) {
      if (auto *F = parseFunction()) TU->Functions.push_back(F);
      skipStatementTerminators();
    }
    if (!at(TokenKind::eof))
      Diag.error(cur().Loc, "stray tokens after function definitions");
  } else {
    // Script file — top-level statements, possibly followed by helper
    // functions and/or classdefs. MATLAB itself requires class files to
    // contain *only* a classdef; we accept script + classdef in one file
    // for testing / single-file demos, which simplifies the run-tests
    // pipeline that compiles one .m at a time.
    auto *S = Ctx.make<Script>();
    S->Body = Ctx.make<Block>();
    S->Range.Begin = cur().Loc;
    while (!at(TokenKind::eof) && !at(TokenKind::kw_function) &&
           !at(TokenKind::kw_classdef)) {
      if (auto *St = parseStmt()) S->Body->Stmts.push_back(St);
      skipStatementTerminators();
    }
    S->Range.End = cur().Loc;
    TU->ScriptNode = S;

    while (at(TokenKind::kw_function) || at(TokenKind::kw_classdef)) {
      if (at(TokenKind::kw_classdef)) {
        if (auto *C = parseClassDef()) TU->Classes.push_back(C);
      } else {
        if (auto *F = parseFunction()) TU->Functions.push_back(F);
      }
      skipStatementTerminators();
    }
  }

  TU->Range.End = cur().Loc;
  return TU;
}

//===----------------------------------------------------------------------===//
// Function
//===----------------------------------------------------------------------===//

Function *Parser::parseFunction() {
  assert(at(TokenKind::kw_function));
  auto *F = Ctx.make<Function>();
  F->Range.Begin = cur().Loc;
  ++Idx; // consume 'function'

  // Optional output list:
  //   function NAME(ARGS)
  //   function OUT = NAME(ARGS)
  //   function [O1, O2] = NAME(ARGS)
  std::vector<std::string_view> Outputs;
  size_t Save = Idx;
  bool HasOutputs = false;
  if (at(TokenKind::l_square)) {
    ++Idx;
    std::vector<std::string_view> Tmp;
    if (!at(TokenKind::r_square)) {
      while (true) {
        if (!at(TokenKind::identifier)) break;
        Tmp.push_back(take().Text);
        if (!consume(TokenKind::comma)) break;
      }
    }
    if (consume(TokenKind::r_square) && consume(TokenKind::equal)) {
      Outputs = std::move(Tmp);
      HasOutputs = true;
    } else {
      Idx = Save; // rewind
    }
  } else if (at(TokenKind::identifier) && peek(1).is(TokenKind::equal)) {
    Outputs.push_back(take().Text);
    ++Idx; // '='
    HasOutputs = true;
  }
  (void)HasOutputs;
  F->Outputs = std::move(Outputs);

  if (!at(TokenKind::identifier)) {
    Diag.error(cur().Loc, "expected function name");
    synchronize();
    return F;
  }
  F->Name = take().Text;
  /* Property accessor method names: `function v = get.Prop(obj)` and
   * `function set.Prop(obj, v)` use a dotted identifier for the
   * method name. Stitch it back together and intern the combined
   * string so downstream Sema / lowering can look it up by name
   * ("get.Prop" / "set.Prop"). Only applies when the head identifier
   * is exactly `get` or `set` to avoid silently turning arbitrary
   * dotted names into valid method names. */
  if ((F->Name == "get" || F->Name == "set") && consume(TokenKind::dot)) {
    if (at(TokenKind::identifier)) {
      std::string Combined = std::string(F->Name) + "." +
                              std::string(take().Text);
      F->Name = Ctx.intern(Combined);
    }
  }

  if (consume(TokenKind::l_paren)) {
    if (!at(TokenKind::r_paren)) {
      while (true) {
        if (at(TokenKind::identifier) || at(TokenKind::tilde)) {
          F->Inputs.push_back(take().Text);
        } else {
          Diag.error(cur().Loc, "expected parameter name");
          break;
        }
        if (!consume(TokenKind::comma)) break;
      }
    }
    expect(TokenKind::r_paren, "')'");
  }
  skipStatementTerminators();

  // Body
  pushScope();
  for (auto N : F->Inputs) addBinding(N);
  for (auto N : F->Outputs) addBinding(N);

  F->Body = Ctx.make<Block>();
  while (!atEndOfBlock() && !at(TokenKind::kw_function)) {
    if (auto *S = parseStmt()) F->Body->Stmts.push_back(S);
    skipStatementTerminators();
  }

  // Nested functions (before the closing 'end', if any).
  while (at(TokenKind::kw_function)) {
    if (auto *N = parseFunction()) F->Nested.push_back(N);
    skipStatementTerminators();
  }

  // Trailing 'end' is optional in script-style function files, but if present
  // we accept and consume it.
  if (at(TokenKind::kw_end)) ++Idx;

  popScope();
  F->Range.End = cur().Loc;
  return F;
}

//===----------------------------------------------------------------------===//
// ClassDef
//
// Parses a minimum MATLAB classdef:
//
//   classdef Name                       % or "classdef Name < Super"
//     properties [(...)]
//       Prop1
//       Prop2 = default
//     end
//     methods [(...)]
//       function obj = Name(args) ... end    % constructor
//       function r = m(obj, ...) ... end
//     end
//   end
//
// Attribute blocks like `(Dependent)`, `(Access=private)` and validation
// `{mustBeNumeric}` are parsed shallowly — we skip to the closing token
// without interpreting them, so they survive round-trip but don't yet
// influence Sema or lowering.
//===----------------------------------------------------------------------===//

ClassDef *Parser::parseClassDef() {
  assert(at(TokenKind::kw_classdef));
  auto *C = Ctx.make<ClassDef>();
  C->Range.Begin = cur().Loc;
  ++Idx; // consume 'classdef'

  /* Optional class-level attribute list: classdef (Attr1, Attr2) Name ...
   * For v1 we skip the whole parenthesised block. */
  if (consume(TokenKind::l_paren)) {
    int Depth = 1;
    while (Depth > 0 && !at(TokenKind::eof)) {
      if (at(TokenKind::l_paren)) ++Depth;
      else if (at(TokenKind::r_paren)) --Depth;
      ++Idx;
    }
  }

  if (!at(TokenKind::identifier)) {
    Diag.error(cur().Loc, "expected class name after 'classdef'");
    synchronize();
    return C;
  }
  C->Name = take().Text;

  /* Optional superclass: `< Super` (or `< A & B`, which we flatten — v1
   * supports single inheritance only, so we record the first super and
   * silently drop any '& extra' for now). */
  if (consume(TokenKind::less)) {
    if (at(TokenKind::identifier)) C->SuperName = take().Text;
    while (consume(TokenKind::amp)) {
      if (at(TokenKind::identifier)) ++Idx; // drop additional supers (v1)
    }
  }
  skipStatementTerminators();

  while (!at(TokenKind::eof) && !at(TokenKind::kw_end)) {
    if (at(TokenKind::kw_properties)) {
      ++Idx;
      /* Parse the optional `(Attr1, Attr2, Key=Val)` attribute block.
       * We capture the subset we understand (Dependent, Constant,
       * Abstract, Access/GetAccess/SetAccess=NAME). Anything else is
       * skipped with depth-tracked paren matching. */
      bool BlockDependent = false;
      bool BlockConstant = false;
      bool BlockAbstract = false;
      std::string_view BlockAccess, BlockGet, BlockSet;
      if (consume(TokenKind::l_paren)) {
        int Depth = 1;
        while (Depth > 0 && !at(TokenKind::eof)) {
          if (at(TokenKind::l_paren)) { ++Depth; ++Idx; continue; }
          if (at(TokenKind::r_paren)) { --Depth; ++Idx; continue; }
          if (at(TokenKind::identifier)) {
            std::string_view K = cur().Text;
            if (K == "Dependent") { BlockDependent = true; ++Idx; continue; }
            if (K == "Constant")  { BlockConstant = true; ++Idx; continue; }
            if (K == "Abstract")  { BlockAbstract = true; ++Idx; continue; }
            if (K == "Access" || K == "GetAccess" || K == "SetAccess") {
              ++Idx;
              if (consume(TokenKind::equal)) {
                if (at(TokenKind::identifier) ||
                    at(TokenKind::string_literal) ||
                    at(TokenKind::char_literal)) {
                  std::string_view V = cur().Text;
                  if (K == "Access")    BlockAccess = V;
                  if (K == "GetAccess") BlockGet = V;
                  if (K == "SetAccess") BlockSet = V;
                  ++Idx;
                  continue;
                }
              }
              continue;
            }
          }
          ++Idx;
        }
      }
      skipStatementTerminators();
      while (!at(TokenKind::kw_end) && !at(TokenKind::eof)) {
        if (!at(TokenKind::identifier)) { ++Idx; continue; }
        ClassProp P;
        P.Range.Begin = cur().Loc;
        P.Name = take().Text;
        P.Dependent = BlockDependent;
        P.Constant  = BlockConstant;
        P.IsAbstract = BlockAbstract;
        P.Access = BlockAccess;
        P.GetAccess = BlockGet;
        P.SetAccess = BlockSet;
        /* Skip optional `(1,1)` size spec or `{mustBeNumeric}` validator. */
        if (consume(TokenKind::l_paren)) {
          int Depth = 1;
          while (Depth > 0 && !at(TokenKind::eof)) {
            if (at(TokenKind::l_paren)) ++Depth;
            else if (at(TokenKind::r_paren)) --Depth;
            ++Idx;
          }
        }
        if (consume(TokenKind::l_brace)) {
          int Depth = 1;
          while (Depth > 0 && !at(TokenKind::eof)) {
            if (at(TokenKind::l_brace)) ++Depth;
            else if (at(TokenKind::r_brace)) --Depth;
            ++Idx;
          }
        }
        /* Optional type annotation, e.g. `Next Node = Node.empty` — drop
         * the `Node` part; we don't type-check it yet. */
        if (at(TokenKind::identifier)) ++Idx;
        if (consume(TokenKind::equal)) {
          P.Default = parseExpr();
        }
        P.Range.End = cur().Loc;
        C->Props.push_back(std::move(P));
        skipStatementTerminators();
      }
      if (at(TokenKind::kw_end)) ++Idx;
      skipStatementTerminators();
    } else if (at(TokenKind::kw_methods)) {
      ++Idx;
      bool IsStatic = false;
      if (consume(TokenKind::l_paren)) {
        /* Scan for the word "Static" inside the attribute list — enough
         * for v1 dispatch. Anything else is skipped. */
        int Depth = 1;
        while (Depth > 0 && !at(TokenKind::eof)) {
          if (at(TokenKind::l_paren)) ++Depth;
          else if (at(TokenKind::r_paren)) --Depth;
          else if (at(TokenKind::identifier) && cur().Text == "Static")
            IsStatic = true;
          ++Idx;
        }
      }
      skipStatementTerminators();
      while (at(TokenKind::kw_function)) {
        if (auto *F = parseFunction()) {
          if (IsStatic) C->StaticMethods.push_back(F);
          else          C->Methods.push_back(F);
        }
        skipStatementTerminators();
      }
      if (at(TokenKind::kw_end)) ++Idx;
      skipStatementTerminators();
    } else if (at(TokenKind::kw_events) ||
               at(TokenKind::kw_enumeration)) {
      /* Parsed-but-ignored: skip to the matching end. */
      ++Idx;
      if (consume(TokenKind::l_paren)) {
        int Depth = 1;
        while (Depth > 0 && !at(TokenKind::eof)) {
          if (at(TokenKind::l_paren)) ++Depth;
          else if (at(TokenKind::r_paren)) --Depth;
          ++Idx;
        }
      }
      while (!at(TokenKind::kw_end) && !at(TokenKind::eof)) ++Idx;
      if (at(TokenKind::kw_end)) ++Idx;
      skipStatementTerminators();
    } else {
      Diag.error(cur().Loc,
                 "unexpected token inside classdef; expected "
                 "properties/methods/end");
      ++Idx;
      skipStatementTerminators();
    }
  }
  if (at(TokenKind::kw_end)) ++Idx;

  C->Range.End = cur().Loc;
  return C;
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

Stmt *Parser::parseStmt() {
  // Skip any leading newlines/separators that survived.
  if (isStatementTerminator() && !at(TokenKind::eof)) {
    ++Idx;
    return nullptr;
  }

  switch (cur().Kind) {
  case TokenKind::kw_if:         return parseIf();
  case TokenKind::kw_for:        return parseFor(/*IsParfor=*/false);
  case TokenKind::kw_parfor:     return parseFor(/*IsParfor=*/true);
  case TokenKind::kw_while:      return parseWhile();
  case TokenKind::kw_switch:     return parseSwitch();
  case TokenKind::kw_try:        return parseTry();
  case TokenKind::kw_global:     return parseGlobal();
  case TokenKind::kw_persistent: return parsePersistent();
  case TokenKind::kw_import:     return parseImport();
  case TokenKind::kw_break:      return parseSimpleStmt(NodeKind::BreakStmt);
  case TokenKind::kw_continue:   return parseSimpleStmt(NodeKind::ContinueStmt);
  case TokenKind::kw_return:     return parseSimpleStmt(NodeKind::ReturnStmt);
  default: break;
  }

  // Detect command syntax at statement start.
  if (auto *Cmd = tryParseCommand()) return Cmd;

  return tryParseAssignOrExprStmt();
}

Stmt *Parser::parseSimpleStmt(NodeKind K) {
  Stmt *S = nullptr;
  SourceLocation Begin = cur().Loc;
  switch (K) {
  case NodeKind::BreakStmt:    S = Ctx.make<BreakStmt>(); break;
  case NodeKind::ContinueStmt: S = Ctx.make<ContinueStmt>(); break;
  case NodeKind::ReturnStmt:   S = Ctx.make<ReturnStmt>(); break;
  default: assert(false);
  }
  S->Range.Begin = Begin;
  ++Idx; // consume keyword
  S->Range.End = cur().Loc;
  return S;
}

GlobalDecl *Parser::parseGlobal() {
  auto *S = Ctx.make<GlobalDecl>();
  S->Range.Begin = cur().Loc;
  ++Idx;
  while (at(TokenKind::identifier)) {
    S->Names.push_back(take().Text);
    addBinding(S->Names.back());
    if (!consume(TokenKind::comma)) break;
  }
  S->Range.End = cur().Loc;
  return S;
}

PersistentDecl *Parser::parsePersistent() {
  auto *S = Ctx.make<PersistentDecl>();
  S->Range.Begin = cur().Loc;
  ++Idx;
  while (at(TokenKind::identifier)) {
    S->Names.push_back(take().Text);
    addBinding(S->Names.back());
    if (!consume(TokenKind::comma)) break;
  }
  S->Range.End = cur().Loc;
  return S;
}

ImportStmt *Parser::parseImport() {
  auto *S = Ctx.make<ImportStmt>();
  S->Range.Begin = cur().Loc;
  ++Idx;
  if (at(TokenKind::identifier)) {
    S->Path.push_back(take().Text);
    while (consume(TokenKind::dot)) {
      if (at(TokenKind::star)) { S->Wildcard = true; ++Idx; break; }
      if (at(TokenKind::identifier)) S->Path.push_back(take().Text);
      else break;
    }
  }
  S->Range.End = cur().Loc;
  return S;
}

IfStmt *Parser::parseIf() {
  auto *N = Ctx.make<IfStmt>();
  N->Range.Begin = cur().Loc;
  ++Idx; // 'if'
  N->Cond = parseExpr();
  skipStatementTerminators();
  N->Then = Ctx.make<Block>();
  while (!atEndOfBlock()) {
    if (auto *S = parseStmt()) N->Then->Stmts.push_back(S);
    skipStatementTerminators();
  }
  while (at(TokenKind::kw_elseif)) {
    ++Idx;
    ElseIf EI;
    EI.Cond = parseExpr();
    skipStatementTerminators();
    EI.Body = Ctx.make<Block>();
    while (!atEndOfBlock()) {
      if (auto *S = parseStmt()) EI.Body->Stmts.push_back(S);
      skipStatementTerminators();
    }
    N->Elseifs.push_back(EI);
  }
  if (consume(TokenKind::kw_else)) {
    skipStatementTerminators();
    N->Else = Ctx.make<Block>();
    while (!atEndOfBlock()) {
      if (auto *S = parseStmt()) N->Else->Stmts.push_back(S);
      skipStatementTerminators();
    }
  }
  expect(TokenKind::kw_end, "'end' to close 'if'");
  N->Range.End = cur().Loc;
  return N;
}

ForStmt *Parser::parseFor(bool IsParfor) {
  auto *N = Ctx.make<ForStmt>();
  N->IsParfor = IsParfor;
  N->Range.Begin = cur().Loc;
  ++Idx; // for / parfor
  // Optional parfor worker count in parentheses is not handled here.
  if (!at(TokenKind::identifier)) {
    Diag.error(cur().Loc, "expected loop variable");
    synchronize();
    return N;
  }
  N->Var = take().Text;
  addBinding(N->Var);
  expect(TokenKind::equal, "'=' in for-loop header");
  N->Iter = parseExpr();
  skipStatementTerminators();
  N->Body = Ctx.make<Block>();
  while (!atEndOfBlock()) {
    if (auto *S = parseStmt()) N->Body->Stmts.push_back(S);
    skipStatementTerminators();
  }
  expect(TokenKind::kw_end, "'end' to close 'for'");
  N->Range.End = cur().Loc;
  return N;
}

WhileStmt *Parser::parseWhile() {
  auto *N = Ctx.make<WhileStmt>();
  N->Range.Begin = cur().Loc;
  ++Idx;
  N->Cond = parseExpr();
  skipStatementTerminators();
  N->Body = Ctx.make<Block>();
  while (!atEndOfBlock()) {
    if (auto *S = parseStmt()) N->Body->Stmts.push_back(S);
    skipStatementTerminators();
  }
  expect(TokenKind::kw_end, "'end' to close 'while'");
  N->Range.End = cur().Loc;
  return N;
}

SwitchStmt *Parser::parseSwitch() {
  auto *N = Ctx.make<SwitchStmt>();
  N->Range.Begin = cur().Loc;
  ++Idx;
  N->Discriminant = parseExpr();
  skipStatementTerminators();
  while (at(TokenKind::kw_case) || at(TokenKind::kw_otherwise)) {
    SwitchCase C;
    if (consume(TokenKind::kw_case)) {
      C.Value = parseExpr();
    } else {
      ++Idx; // otherwise
      C.Value = nullptr;
    }
    skipStatementTerminators();
    C.Body = Ctx.make<Block>();
    while (!atEndOfBlock()) {
      if (auto *S = parseStmt()) C.Body->Stmts.push_back(S);
      skipStatementTerminators();
    }
    N->Cases.push_back(C);
  }
  expect(TokenKind::kw_end, "'end' to close 'switch'");
  N->Range.End = cur().Loc;
  return N;
}

TryStmt *Parser::parseTry() {
  auto *N = Ctx.make<TryStmt>();
  N->Range.Begin = cur().Loc;
  ++Idx;
  skipStatementTerminators();
  N->TryBody = Ctx.make<Block>();
  while (!atEndOfBlock()) {
    if (auto *S = parseStmt()) N->TryBody->Stmts.push_back(S);
    skipStatementTerminators();
  }
  if (consume(TokenKind::kw_catch)) {
    if (at(TokenKind::identifier) && !peek(1).is(TokenKind::equal)) {
      // heuristic: identifier on same line as 'catch' binds the exception
      N->CatchVar = take().Text;
      addBinding(N->CatchVar);
    }
    skipStatementTerminators();
    N->CatchBody = Ctx.make<Block>();
    while (!atEndOfBlock()) {
      if (auto *S = parseStmt()) N->CatchBody->Stmts.push_back(S);
      skipStatementTerminators();
    }
  }
  expect(TokenKind::kw_end, "'end' to close 'try'");
  N->Range.End = cur().Loc;
  return N;
}

//===----------------------------------------------------------------------===//
// Command syntax
//===----------------------------------------------------------------------===//

// Returns true if this token, appearing in "ident SPACE <tok>" at statement
// start, signals that we should treat the remainder as bare-word arguments.
static bool couldStartCommandArg(const Token &T) {
  switch (T.Kind) {
  case TokenKind::identifier:
  case TokenKind::integer_literal:
  case TokenKind::float_literal:
  case TokenKind::imag_literal:
  case TokenKind::string_literal:
  case TokenKind::char_literal:
    return true;
  default:
    return false;
  }
}

CommandStmt *Parser::tryParseCommand() {
  if (!at(TokenKind::identifier)) return nullptr;
  if (!cur().StartsStmt) return nullptr;

  std::string_view Name = cur().Text;
  if (isBound(Name)) return nullptr;

  const Token &Next = peek(1);
  if (!Next.LeadingSpace) return nullptr;
  if (!couldStartCommandArg(Next)) return nullptr;

  // Heuristic: if the next token is an identifier followed by `=` (likely
  // an assignment like `x = 1`), not a command. The peek is "ident SPACE ident"
  // — if the identifier after is followed by an operator that typically
  // continues an expression with the first identifier (like `+`, `-`), it's
  // ambiguous. MATLAB resolves this at runtime; we use the statement-start
  // plus leading-space signal conservatively.
  //
  // Specifically, reject if the *third* token is '=' on its own (assignment),
  // or a binary operator (arithmetic continuation) without surrounding spaces.
  const Token &After = peek(2);
  if (After.is(TokenKind::equal) && !peek(3).is(TokenKind::equal))
    return nullptr;

  auto *Cmd = Ctx.make<CommandStmt>();
  Cmd->Range.Begin = cur().Loc;
  Cmd->Name = Name;
  ++Idx; // consume command name

  // Collect bare-word / literal arguments until statement terminator.
  while (!isStatementTerminator()) {
    const Token &T = cur();
    std::string Arg;
    if (T.is(TokenKind::string_literal)) {
      Arg = unescapeDoubleQuoted(T.Text);
    } else if (T.is(TokenKind::char_literal)) {
      Arg = unescapeSingleQuoted(T.Text);
    } else {
      Arg = std::string(T.Text);
    }
    Cmd->Args.push_back(std::move(Arg));
    ++Idx;
  }
  Cmd->Suppressed = at(TokenKind::semi);
  Cmd->Range.End = cur().Loc;
  return Cmd;
}

//===----------------------------------------------------------------------===//
// Assignment vs expression statement
//===----------------------------------------------------------------------===//

Stmt *Parser::tryParseAssignOrExprStmt() {
  SourceLocation Begin = cur().Loc;

  // Special-case multi-assignment LHS: `[a, b, c] = rhs`
  if (at(TokenKind::l_square)) {
    size_t Save = Idx;
    ++Idx;
    std::vector<Expr *> Targets;
    if (!at(TokenKind::r_square)) {
      while (true) {
        Expr *E = parsePostfix(parsePrimary());
        Targets.push_back(E);
        if (!consume(TokenKind::comma)) break;
      }
    }
    if (consume(TokenKind::r_square) && consume(TokenKind::equal)) {
      auto *A = Ctx.make<AssignStmt>();
      A->Range.Begin = Begin;
      A->LHS = std::move(Targets);
      for (auto *T : A->LHS) {
        if (auto *N = dynamic_cast<NameExpr *>(T))
          addBinding(N->Name);
      }
      A->RHS = parseExpr();
      A->Suppressed = at(TokenKind::semi);
      A->Range.End = cur().Loc;
      return A;
    }
    // Not an assignment — rewind.
    Idx = Save;
  }

  // Normal: parse an expression, then check for '='.
  Expr *E = parseExpr();
  if (consume(TokenKind::equal)) {
    if (!lhsIsAssignable(E))
      Diag.error(E->Range.Begin, "expression is not assignable");
    auto *A = Ctx.make<AssignStmt>();
    A->Range.Begin = Begin;
    A->LHS.push_back(E);
    if (auto *N = dynamic_cast<NameExpr *>(E))
      addBinding(N->Name);
    A->RHS = parseExpr();
    A->Suppressed = at(TokenKind::semi);
    A->Range.End = cur().Loc;
    return A;
  }

  auto *ES = Ctx.make<ExprStmt>();
  ES->Range.Begin = Begin;
  ES->E = E;
  ES->Suppressed = at(TokenKind::semi);
  ES->Range.End = cur().Loc;
  return ES;
}

bool Parser::lhsIsAssignable(const Expr *E) const {
  if (!E) return false;
  switch (E->Kind) {
  case NodeKind::NameExpr:
  case NodeKind::CallOrIndex:      // a(i) = x
  case NodeKind::CellIndex:        // c{i} = x
  case NodeKind::FieldAccess:      // s.x = v
  case NodeKind::DynamicField:     // s.(n) = v
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Expressions — Pratt parser
//===----------------------------------------------------------------------===//

namespace {
struct BinPrec {
  int Prec;          // -1 if not a binary operator
  bool RightAssoc;
  BinOp Op;
};

BinPrec binOpFor(TokenKind K) {
  switch (K) {
  case TokenKind::pipe_pipe:   return {1,  false, BinOp::ShortOr};
  case TokenKind::amp_amp:     return {2,  false, BinOp::ShortAnd};
  case TokenKind::pipe:        return {3,  false, BinOp::Or};
  case TokenKind::amp:         return {4,  false, BinOp::And};
  case TokenKind::less:        return {5,  false, BinOp::Lt};
  case TokenKind::less_eq:     return {5,  false, BinOp::Le};
  case TokenKind::greater:     return {5,  false, BinOp::Gt};
  case TokenKind::greater_eq:  return {5,  false, BinOp::Ge};
  case TokenKind::eq_eq:       return {5,  false, BinOp::Eq};
  case TokenKind::bang_eq:     return {5,  false, BinOp::Ne};
  case TokenKind::plus:        return {7,  false, BinOp::Add};
  case TokenKind::minus:       return {7,  false, BinOp::Sub};
  case TokenKind::star:        return {8,  false, BinOp::Mul};
  case TokenKind::slash:       return {8,  false, BinOp::Div};
  case TokenKind::backslash:   return {8,  false, BinOp::LeftDiv};
  case TokenKind::dot_star:    return {8,  false, BinOp::ElemMul};
  case TokenKind::dot_slash:   return {8,  false, BinOp::ElemDiv};
  case TokenKind::dot_backslash:return{8, false, BinOp::ElemLeftDiv};
  case TokenKind::caret:       return {10, true,  BinOp::Pow};
  case TokenKind::dot_caret:   return {10, true,  BinOp::ElemPow};
  default:                     return {-1, false, BinOp::Add};
  }
}
} // namespace

Expr *Parser::parseExpr() { return parseRange(); }

Expr *Parser::parseRange() {
  Expr *Start = parseBinary(1);
  if (!consume(TokenKind::colon)) return Start;

  // a:b[:c]
  auto *R = Ctx.make<RangeExpr>();
  R->Range.Begin = Start ? Start->Range.Begin : cur().Loc;
  R->Start = Start;
  Expr *Mid = parseBinary(1);
  if (consume(TokenKind::colon)) {
    R->Step = Mid;
    R->End = parseBinary(1);
  } else {
    R->End = Mid;
  }
  R->Range.End = cur().Loc;
  return R;
}

Expr *Parser::parseBinary(int MinPrec) {
  Expr *LHS = parseUnary();
  while (true) {
    BinPrec BP = binOpFor(cur().Kind);
    if (BP.Prec < MinPrec) break;
    Token OpTok = take();
    int NextMin = BP.RightAssoc ? BP.Prec : BP.Prec + 1;
    Expr *RHS = parseBinary(NextMin);
    auto *B = Ctx.make<BinaryOpExpr>();
    B->Op = BP.Op;
    B->LHS = LHS;
    B->RHS = RHS;
    B->Range.Begin = LHS->Range.Begin;
    B->Range.End = RHS ? RHS->Range.End : OpTok.endLoc();
    LHS = B;
  }
  return LHS;
}

Expr *Parser::parseUnary() {
  if (at(TokenKind::plus) || at(TokenKind::minus) || at(TokenKind::tilde)) {
    Token OpTok = take();
    UnOp Op = (OpTok.Kind == TokenKind::plus)  ? UnOp::Plus
            : (OpTok.Kind == TokenKind::minus) ? UnOp::Minus
                                                : UnOp::Not;
    Expr *Inner = parseUnary();
    auto *U = Ctx.make<UnaryOpExpr>();
    U->Op = Op;
    U->Operand = Inner;
    U->Range.Begin = OpTok.Loc;
    U->Range.End = Inner ? Inner->Range.End : OpTok.endLoc();
    return U;
  }
  return parsePostfix(parsePrimary());
}

Expr *Parser::parsePostfix(Expr *LHS) {
  while (true) {
    if (at(TokenKind::l_paren)) {
      ++Idx;
      ++IndexDepth;
      auto *C = Ctx.make<CallOrIndex>();
      C->Callee = LHS;
      C->Args = parseArgList(TokenKind::r_paren);
      --IndexDepth;
      expect(TokenKind::r_paren, "')'");
      C->Range.Begin = LHS->Range.Begin;
      C->Range.End = cur().Loc;
      LHS = C;
      continue;
    }
    if (at(TokenKind::l_brace)) {
      ++Idx;
      ++IndexDepth;
      auto *C = Ctx.make<CellIndex>();
      C->Callee = LHS;
      C->Args = parseArgList(TokenKind::r_brace);
      --IndexDepth;
      expect(TokenKind::r_brace, "'}'");
      C->Range.Begin = LHS->Range.Begin;
      C->Range.End = cur().Loc;
      LHS = C;
      continue;
    }
    if (at(TokenKind::dot)) {
      ++Idx;
      if (at(TokenKind::l_paren)) {
        ++Idx;
        auto *D = Ctx.make<DynamicField>();
        D->Base = LHS;
        D->Name = parseExpr();
        expect(TokenKind::r_paren, "')'");
        D->Range.Begin = LHS->Range.Begin;
        D->Range.End = cur().Loc;
        LHS = D;
      } else if (at(TokenKind::identifier)) {
        auto *F = Ctx.make<FieldAccess>();
        F->Base = LHS;
        F->Field = take().Text;
        F->Range.Begin = LHS->Range.Begin;
        F->Range.End = cur().Loc;
        LHS = F;
      } else {
        Diag.error(cur().Loc, "expected field name after '.'");
        break;
      }
      continue;
    }
    if (at(TokenKind::apostrophe) || at(TokenKind::dot_apostrophe)) {
      auto *P = Ctx.make<PostfixOpExpr>();
      P->Op = at(TokenKind::apostrophe) ? PostfixOp::CTranspose
                                        : PostfixOp::Transpose;
      P->Operand = LHS;
      P->Range.Begin = LHS->Range.Begin;
      P->Range.End = cur().endLoc();
      ++Idx;
      LHS = P;
      continue;
    }
    break;
  }
  return LHS;
}

Expr *Parser::parsePrimary() {
  const Token &T = cur();
  switch (T.Kind) {
  case TokenKind::integer_literal: {
    auto *L = Ctx.make<IntegerLiteral>();
    L->Text = T.Text;
    L->Range = T.range();
    ++Idx;
    return L;
  }
  case TokenKind::float_literal: {
    auto *L = Ctx.make<FPLiteral>();
    L->Text = T.Text;
    L->Range = T.range();
    ++Idx;
    return L;
  }
  case TokenKind::imag_literal: {
    auto *L = Ctx.make<ImagLiteral>();
    L->Text = T.Text;
    L->Range = T.range();
    ++Idx;
    return L;
  }
  case TokenKind::string_literal: {
    auto *L = Ctx.make<StringLiteral>();
    L->Value = unescapeDoubleQuoted(T.Text);
    L->Range = T.range();
    ++Idx;
    return L;
  }
  case TokenKind::char_literal: {
    auto *L = Ctx.make<CharLiteral>();
    L->Value = unescapeSingleQuoted(T.Text);
    L->Range = T.range();
    ++Idx;
    return L;
  }
  case TokenKind::identifier: {
    auto *N = Ctx.make<NameExpr>();
    N->Name = T.Text;
    N->Range = T.range();
    ++Idx;
    return N;
  }
  case TokenKind::kw_end: {
    if (IndexDepth == 0) {
      Diag.error(T.Loc, "'end' is only valid inside indexing");
      ++Idx;
      auto *E = Ctx.make<EndExpr>();
      E->Range = T.range();
      return E;
    }
    ++Idx;
    auto *E = Ctx.make<EndExpr>();
    E->Range = T.range();
    return E;
  }
  case TokenKind::colon: {
    // bare `:` inside an argument list means "all"
    auto *C = Ctx.make<ColonExpr>();
    C->Range = T.range();
    ++Idx;
    return C;
  }
  case TokenKind::l_paren:    return parseParenExpr();
  case TokenKind::l_square:   return parseMatrixOrCell(/*IsCell=*/false);
  case TokenKind::l_brace:    return parseMatrixOrCell(/*IsCell=*/true);
  case TokenKind::at:         return parseAnonOrHandle();
  default:
    Diag.error(T.Loc, std::string("unexpected '") + std::string(T.Text) +
                          "' in expression");
    auto *E = Ctx.make<NameExpr>();
    E->Name = "<error>";
    E->Range = T.range();
    ++Idx;
    return E;
  }
}

Expr *Parser::parseParenExpr() {
  ++Idx; // '('
  Expr *E = parseExpr();
  expect(TokenKind::r_paren, "')'");
  return E;
}

Expr *Parser::parseAnonOrHandle() {
  SourceLocation Begin = cur().Loc;
  ++Idx; // '@'
  if (at(TokenKind::l_paren)) {
    auto *A = Ctx.make<AnonFunction>();
    A->Range.Begin = Begin;
    ++Idx;
    if (!at(TokenKind::r_paren)) {
      while (true) {
        if (!at(TokenKind::identifier)) {
          Diag.error(cur().Loc, "expected parameter name");
          break;
        }
        A->Params.push_back(take().Text);
        if (!consume(TokenKind::comma)) break;
      }
    }
    expect(TokenKind::r_paren, "')'");
    pushScope();
    for (auto N : A->Params) addBinding(N);
    A->Body = parseExpr();
    popScope();
    A->Range.End = cur().Loc;
    return A;
  }
  if (at(TokenKind::identifier)) {
    auto *F = Ctx.make<FuncHandle>();
    F->Name = take().Text;
    F->Range.Begin = Begin;
    F->Range.End = cur().Loc;
    return F;
  }
  Diag.error(cur().Loc, "expected function name or '(' after '@'");
  auto *F = Ctx.make<FuncHandle>();
  F->Name = "<error>";
  F->Range.Begin = Begin;
  F->Range.End = cur().Loc;
  return F;
}

//===----------------------------------------------------------------------===//
// Matrix / cell literals
//===----------------------------------------------------------------------===//

// In a matrix-literal row, whitespace separates elements. So
//   [1 -2]  -> two elements (1, -2)
//   [1-2]   -> one element (-1)
// We implement this by asking: when we see a +/- with no space before but
// space after (or similar asymmetry), it's an infix operator. When there is
// a space before and the next token starts a new primary immediately, it's
// a unary sign introducing a new element.
//
// Concretely, after parsing an element we decide whether to treat the next
// +/- as a new unary-led element:
//   - if the +/- has LeadingSpace=true AND the token after +/- has
//     LeadingSpace=false, it's a unary sign (new element).
//   - otherwise it's a binary operator in the current expression.

static bool looksLikeNewUnarySign(const Token &Op, const Token &AfterOp) {
  if (Op.Kind != TokenKind::plus && Op.Kind != TokenKind::minus) return false;
  if (!Op.LeadingSpace) return false;       // no space before => binary
  if (AfterOp.LeadingSpace) return false;   // space before AND after => binary
  return true;
}

static bool rowTerminator(TokenKind K) {
  return K == TokenKind::semi || K == TokenKind::newline ||
         K == TokenKind::r_square || K == TokenKind::r_brace ||
         K == TokenKind::eof;
}

std::vector<Expr *> Parser::parseMatrixRow(bool IsCell) {
  (void)IsCell;
  std::vector<Expr *> Row;
  if (rowTerminator(cur().Kind)) return Row;

  // A custom Pratt loop that aborts on "unary sign" boundaries.
  auto parseElement = [&]() -> Expr * {
    // Start with a unary/primary/postfix expression.
    Expr *LHS = parseUnary();
    while (true) {
      BinPrec BP = binOpFor(cur().Kind);
      if (BP.Prec < 1) break;
      // Check for +/- that should be a new element instead.
      if (looksLikeNewUnarySign(cur(), peek(1))) break;
      Token OpTok = take();
      int NextMin = BP.RightAssoc ? BP.Prec : BP.Prec + 1;
      // Inner binary parse can use normal precedence.
      Expr *RHS = parseBinary(NextMin);
      auto *B = Ctx.make<BinaryOpExpr>();
      B->Op = BP.Op;
      B->LHS = LHS;
      B->RHS = RHS;
      B->Range.Begin = LHS->Range.Begin;
      B->Range.End = RHS ? RHS->Range.End : OpTok.endLoc();
      LHS = B;
    }
    // Range operator `:` also allowed within matrix rows.
    if (consume(TokenKind::colon)) {
      auto *R = Ctx.make<RangeExpr>();
      R->Range.Begin = LHS->Range.Begin;
      R->Start = LHS;
      Expr *Mid = parseBinary(1);
      if (consume(TokenKind::colon)) {
        R->Step = Mid;
        R->End = parseBinary(1);
      } else {
        R->End = Mid;
      }
      R->Range.End = cur().Loc;
      LHS = R;
    }
    return LHS;
  };

  Row.push_back(parseElement());
  while (!rowTerminator(cur().Kind)) {
    // Commas are explicit separators.
    if (consume(TokenKind::comma)) {
      Row.push_back(parseElement());
      continue;
    }
    // Otherwise, there must be whitespace between elements.
    if (!cur().LeadingSpace) break;
    Row.push_back(parseElement());
  }
  return Row;
}

Expr *Parser::parseMatrixOrCell(bool IsCell) {
  TokenKind Open = IsCell ? TokenKind::l_brace : TokenKind::l_square;
  TokenKind Close = IsCell ? TokenKind::r_brace : TokenKind::r_square;
  SourceLocation Begin = cur().Loc;
  assert(cur().Kind == Open);
  ++Idx;

  std::vector<std::vector<Expr *>> Rows;

  // Skip leading newlines/semicolons (empty rows).
  while (consume(TokenKind::newline) || consume(TokenKind::semi))
    ;

  if (!at(Close)) {
    while (true) {
      auto Row = parseMatrixRow(IsCell);
      if (!Row.empty()) Rows.push_back(std::move(Row));
      // row separator
      bool Sep = false;
      while (consume(TokenKind::newline) || consume(TokenKind::semi)) Sep = true;
      if (at(Close)) break;
      if (!Sep && !at(Close)) {
        // syntax error — try to recover by advancing one token
        Diag.error(cur().Loc, "expected ';' or newline between matrix rows");
        if (!rowTerminator(cur().Kind)) ++Idx;
      }
    }
  }
  expect(Close, IsCell ? "'}'" : "']'");

  Expr *Out;
  if (IsCell) {
    auto *M = Ctx.make<CellLiteral>();
    M->Rows = std::move(Rows);
    M->Range.Begin = Begin;
    M->Range.End = cur().Loc;
    Out = M;
  } else {
    auto *M = Ctx.make<MatrixLiteral>();
    M->Rows = std::move(Rows);
    M->Range.Begin = Begin;
    M->Range.End = cur().Loc;
    Out = M;
  }
  return Out;
}

//===----------------------------------------------------------------------===//
// Argument lists
//===----------------------------------------------------------------------===//

std::vector<Expr *> Parser::parseArgList(TokenKind Closer) {
  std::vector<Expr *> Args;
  if (at(Closer)) return Args;
  while (true) {
    Args.push_back(parseExpr());
    if (!consume(TokenKind::comma)) break;
  }
  return Args;
}

} // namespace matlab
