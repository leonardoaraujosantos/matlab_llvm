#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Lex/Token.h"

#include <string_view>
#include <unordered_set>
#include <vector>

namespace matlab {

class Parser {
public:
  Parser(std::vector<Token> Tokens, ASTContext &Ctx, DiagnosticEngine &Diag);

  /// Parse a whole `.m` file. The default form distinguishes a script
  /// file, a function file, and a (single) classdef file — matching
  /// MATLAB's one-classdef-per-file rule.
  ///
  /// `AllowMultipleUnits` relaxes that rule for **prelude buffers** — a
  /// machine-assembled concatenation of several toolbox classdefs +
  /// helper functions (no script body), parsed standalone. Without it,
  /// such a buffer errors with "stray tokens after classdef" the moment
  /// a second classdef appears, and the caller drops the entire prelude
  /// (see the `-dap` classdef-prelude merge in tools/matlabc). It does
  /// not affect normal file parsing.
  TranslationUnit *parseFile(bool AllowMultipleUnits = false);

private:
  std::vector<Token> Toks;
  size_t Idx = 0;
  ASTContext &Ctx;
  DiagnosticEngine &Diag;

  // Tracks which names have been assigned in the current scope, used to
  // disambiguate command syntax (`disp hello` vs `a + 1`).
  std::vector<std::unordered_set<std::string_view>> ScopeStack;

  // Depth of open indexing contexts (paren after a postfix-able expr,
  // or brace cell-indexing). `end` is only a valid expression when > 0.
  unsigned IndexDepth = 0;

  //--- token helpers
  const Token &peek(size_t k = 0) const { return Toks[Idx + k]; }
  const Token &cur() const { return Toks[Idx]; }
  bool at(TokenKind K) const { return cur().is(K); }
  bool consume(TokenKind K) {
    if (!at(K)) return false;
    ++Idx;
    return true;
  }
  bool expect(TokenKind K, const char *What);
  Token take() { return Toks[Idx++]; }

  void skipStatementTerminators();
  bool isStatementTerminator() const;
  bool atEndOfBlock() const;

  void pushScope() { ScopeStack.emplace_back(); }
  void popScope() { ScopeStack.pop_back(); }
  void addBinding(std::string_view Name) {
    if (!ScopeStack.empty()) ScopeStack.back().insert(Name);
  }
  bool isBound(std::string_view Name) const {
    for (auto It = ScopeStack.rbegin(); It != ScopeStack.rend(); ++It)
      if (It->count(Name)) return true;
    return false;
  }

  //--- top level
  Function *parseFunction();
  ClassDef *parseClassDef();
  Block *parseBlock(bool (Parser::*Stop)() const);

  //--- statements
  Stmt *parseStmt();
  Stmt *tryParseAssignOrExprStmt();
  IfStmt *parseIf();
  ForStmt *parseFor(bool IsParfor);
  WhileStmt *parseWhile();
  SwitchStmt *parseSwitch();
  TryStmt *parseTry();
  GlobalDecl *parseGlobal();
  PersistentDecl *parsePersistent();
  ImportStmt *parseImport();
  Stmt *parseSimpleStmt(NodeKind K); // for break/continue/return

  // Command-syntax detection at statement start.
  CommandStmt *tryParseCommand();

  //--- expressions (Pratt)
  Expr *parseExpr();
  Expr *parseRange();
  Expr *parseBinary(int MinPrec);
  Expr *parseUnary();
  Expr *parsePostfix(Expr *LHS);
  Expr *parsePrimary();

  Expr *parseMatrixOrCell(bool IsCell);
  Expr *parseAnonOrHandle();
  Expr *parseParenExpr();

  // Parse a comma/whitespace-separated list of expressions for matrix/cell
  // literals, honoring MATLAB's whitespace-sensitive sign rules.
  std::vector<Expr *> parseMatrixRow(bool IsCell);

  // Argument lists for calls/indexing: allows bare `:` and `end`.
  std::vector<Expr *> parseArgList(TokenKind Closer);

  //--- misc
  bool lhsIsAssignable(const Expr *E) const;
  void collectLValueNames(const Expr *E);
  void synchronize();

  // Unescape "..." / '...' literal contents into a std::string.
  static std::string unescapeDoubleQuoted(std::string_view Raw);
  static std::string unescapeSingleQuoted(std::string_view Raw);
};

} // namespace matlab
