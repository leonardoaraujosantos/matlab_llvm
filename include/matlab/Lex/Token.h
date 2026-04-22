#pragma once

#include "matlab/Basic/SourceManager.h"

#include <string_view>

namespace matlab {

enum class TokenKind : uint16_t {
#define TOK(Name, Spelling) Name,
#include "matlab/Lex/TokenKinds.def"
  NUM_TOKENS
};

const char *tokenName(TokenKind K);      // spelling from the .def file
const char *tokenKindName(TokenKind K);  // enum-like identifier (for dumps)

struct Token {
  TokenKind Kind = TokenKind::unknown;
  SourceLocation Loc;
  uint32_t Length = 0;
  std::string_view Text; // view into the source buffer

  // Flags useful to the parser.
  bool LeadingSpace = false;   // was there whitespace immediately before?
  bool StartsLine = false;     // first non-whitespace on its line?
  bool StartsStmt = false;     // first token of a statement (after ;, ,, newline)

  bool is(TokenKind K) const { return Kind == K; }
  bool isOneOf(std::initializer_list<TokenKind> Ks) const {
    for (auto K : Ks) if (Kind == K) return true;
    return false;
  }

  SourceLocation endLoc() const {
    return {Loc.File, Loc.Offset + Length};
  }
  SourceRange range() const { return {Loc, endLoc()}; }
};

} // namespace matlab
