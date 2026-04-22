#pragma once

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Token.h"

#include <string_view>
#include <vector>

namespace matlab {

/// Context-sensitive MATLAB lexer.
///
/// Produces a flat token stream including explicit `newline` tokens (MATLAB
/// treats newlines as statement terminators when not escaped by `...`).
/// Comments and line continuations are absorbed and not returned.
class Lexer {
public:
  Lexer(const SourceManager &SM, FileID File, DiagnosticEngine &Diag);

  // Tokenize the whole buffer in one go. Always emits a trailing `eof` token.
  std::vector<Token> tokenize();

private:
  FileID File;
  DiagnosticEngine &Diag;
  std::string_view Src;
  uint32_t Pos = 0;

  // Flags tracked across the stream.
  TokenKind PrevKind = TokenKind::unknown; // kind of previous *emitted* token
  bool AtLineStart = true;
  bool PendingLeadingSpace = false;
  bool StmtStart = true;

  Token next();
  Token makeToken(TokenKind K, uint32_t Begin, uint32_t End);

  void skipHorizontalSpaceAndLineContinuations();
  bool tryLineComment();            // '%' to end-of-line (not block)
  bool tryBlockComment();           // '%{' ... '%}' on their own lines

  Token lexIdentifierOrKeyword(uint32_t Begin);
  Token lexNumber(uint32_t Begin);
  Token lexString(uint32_t Begin);  // "..."
  Token lexChar(uint32_t Begin);    // '...'
  Token lexPunct(uint32_t Begin);

  bool prevAllowsTranspose() const;

  // Lookahead helpers
  char peek(uint32_t k = 0) const {
    return (Pos + k < Src.size()) ? Src[Pos + k] : '\0';
  }
  bool eat(char c) {
    if (peek() == c) { ++Pos; return true; }
    return false;
  }
  bool eof() const { return Pos >= Src.size(); }
};

} // namespace matlab
