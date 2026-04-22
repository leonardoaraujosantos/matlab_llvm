#include "matlab/Lex/Lexer.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <string>
#include <unordered_map>

namespace matlab {

//===----------------------------------------------------------------------===//
// Token name tables
//===----------------------------------------------------------------------===//

static const char *SpellingTable[] = {
#define TOK(Name, Spelling) Spelling,
#include "matlab/Lex/TokenKinds.def"
};

static const char *KindNameTable[] = {
#define TOK(Name, Spelling) #Name,
#include "matlab/Lex/TokenKinds.def"
};

const char *tokenName(TokenKind K) {
  unsigned I = static_cast<unsigned>(K);
  if (I >= static_cast<unsigned>(TokenKind::NUM_TOKENS))
    return "<invalid>";
  return SpellingTable[I];
}
const char *tokenKindName(TokenKind K) {
  unsigned I = static_cast<unsigned>(K);
  if (I >= static_cast<unsigned>(TokenKind::NUM_TOKENS))
    return "<invalid>";
  return KindNameTable[I];
}

//===----------------------------------------------------------------------===//
// Keyword table
//===----------------------------------------------------------------------===//

static const std::unordered_map<std::string_view, TokenKind> &keywordTable() {
  static const std::unordered_map<std::string_view, TokenKind> T = {
#define TOK(Name, Spelling)
#define KW(Name, Spelling) {Spelling, TokenKind::kw_##Name},
#include "matlab/Lex/TokenKinds.def"
  };
  return T;
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(const SourceManager &SM, FileID File, DiagnosticEngine &Diag)
    : File(File), Diag(Diag), Src(SM.getBuffer(File)) {}

Token Lexer::makeToken(TokenKind K, uint32_t Begin, uint32_t End) {
  Token T;
  T.Kind = K;
  T.Loc = {File, Begin};
  T.Length = End - Begin;
  T.Text = Src.substr(Begin, T.Length);
  T.LeadingSpace = PendingLeadingSpace;
  T.StartsLine = AtLineStart;
  T.StartsStmt = StmtStart;
  return T;
}

static bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
static bool isIdentCont(char c)  { return std::isalnum((unsigned char)c) || c == '_'; }

void Lexer::skipHorizontalSpaceAndLineContinuations() {
  while (!eof()) {
    char c = peek();
    if (c == ' ' || c == '\t' || c == '\r') {
      ++Pos;
      PendingLeadingSpace = true;
      continue;
    }
    // Line continuation: '...' followed by optional trailing chars until newline.
    if (c == '.' && peek(1) == '.' && peek(2) == '.') {
      Pos += 3;
      while (!eof() && peek() != '\n') ++Pos;
      if (!eof() && peek() == '\n') ++Pos;
      PendingLeadingSpace = true;
      // note: we do NOT reset AtLineStart — logically still same line
      continue;
    }
    // Block comment: '%{' at the start of a line, paired with '%}' likewise.
    if (AtLineStart && c == '%' && peek(1) == '{') {
      if (tryBlockComment()) continue;
    }
    // Line comment: '%' or '%%' up to end-of-line.
    if (c == '%') {
      tryLineComment();
      // fall through so the next char (newline or EOF) is handled by the caller
      continue;
    }
    if (c == '#') {
      // MATLAB also accepts '#' as a line comment in some dialects; treat same.
      tryLineComment();
      continue;
    }
    break;
  }
}

bool Lexer::tryLineComment() {
  if (peek() != '%' && peek() != '#') return false;
  while (!eof() && peek() != '\n') ++Pos;
  return true;
}

bool Lexer::tryBlockComment() {
  // Preconditions: at line start, peek() == '%', peek(1) == '{'
  assert(peek() == '%' && peek(1) == '{');
  uint32_t Begin = Pos;
  Pos += 2;
  // '%{' must be alone on its line (after optional whitespace). If not, treat
  // as a line comment starting at the original '%'.
  while (!eof() && (peek() == ' ' || peek() == '\t' || peek() == '\r'))
    ++Pos;
  if (!eof() && peek() != '\n') {
    // Not a valid block-comment opener; rewind and fall through as line comment.
    Pos = Begin;
    return false;
  }
  if (!eof()) ++Pos; // consume '\n'
  // Scan until we see '%}' alone on a line.
  while (!eof()) {
    // consume leading whitespace of this line
    uint32_t LineBegin = Pos;
    while (!eof() && (peek() == ' ' || peek() == '\t')) ++Pos;
    if (peek() == '%' && peek(1) == '}') {
      Pos += 2;
      // tolerate trailing space/comment, then eat newline
      while (!eof() && peek() != '\n') ++Pos;
      if (!eof()) ++Pos;
      AtLineStart = true;
      PendingLeadingSpace = true;
      return true;
    }
    // skip to end of line
    Pos = LineBegin;
    while (!eof() && peek() != '\n') ++Pos;
    if (!eof()) ++Pos;
  }
  Diag.error(SourceLocation{File, Begin}, "unterminated block comment");
  return true;
}

bool Lexer::prevAllowsTranspose() const {
  switch (PrevKind) {
  case TokenKind::identifier:
  case TokenKind::integer_literal:
  case TokenKind::float_literal:
  case TokenKind::imag_literal:
  case TokenKind::r_paren:
  case TokenKind::r_square:
  case TokenKind::r_brace:
  case TokenKind::kw_end:
  case TokenKind::apostrophe:
  case TokenKind::dot_apostrophe:
  case TokenKind::dot:
    return true;
  default:
    return false;
  }
}

Token Lexer::lexIdentifierOrKeyword(uint32_t Begin) {
  while (!eof() && isIdentCont(peek())) ++Pos;
  std::string_view Text = Src.substr(Begin, Pos - Begin);
  const auto &KW = keywordTable();
  auto It = KW.find(Text);
  TokenKind K = (It != KW.end()) ? It->second : TokenKind::identifier;
  return makeToken(K, Begin, Pos);
}

Token Lexer::lexNumber(uint32_t Begin) {
  bool IsFloat = false;

  // Hex or binary prefix
  if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
    Pos += 2;
    while (!eof() && std::isxdigit((unsigned char)peek())) ++Pos;
  } else if (peek() == '0' && (peek(1) == 'b' || peek(1) == 'B')) {
    Pos += 2;
    while (!eof() && (peek() == '0' || peek() == '1')) ++Pos;
  } else {
    while (!eof() && std::isdigit((unsigned char)peek())) ++Pos;
    if (peek() == '.' && std::isdigit((unsigned char)peek(1))) {
      IsFloat = true;
      ++Pos;
      while (!eof() && std::isdigit((unsigned char)peek())) ++Pos;
    } else if (peek() == '.' && !isIdentStart(peek(1)) && peek(1) != '.') {
      // "3." with no fractional digits is still a float (MATLAB allows this)
      IsFloat = true;
      ++Pos;
    }
    if (peek() == 'e' || peek() == 'E') {
      IsFloat = true;
      ++Pos;
      if (peek() == '+' || peek() == '-') ++Pos;
      while (!eof() && std::isdigit((unsigned char)peek())) ++Pos;
    }
  }

  TokenKind K = IsFloat ? TokenKind::float_literal : TokenKind::integer_literal;

  // Imaginary suffix: 'i' or 'j' (must not be followed by identifier char).
  if ((peek() == 'i' || peek() == 'j') && !isIdentCont(peek(1))) {
    ++Pos;
    K = TokenKind::imag_literal;
  }

  return makeToken(K, Begin, Pos);
}

Token Lexer::lexString(uint32_t Begin) {
  // Consume opening '"'
  ++Pos;
  while (!eof()) {
    char c = peek();
    if (c == '"') {
      if (peek(1) == '"') { Pos += 2; continue; } // escaped quote
      ++Pos;
      return makeToken(TokenKind::string_literal, Begin, Pos);
    }
    if (c == '\n') {
      Diag.error(SourceLocation{File, Begin}, "unterminated string literal");
      return makeToken(TokenKind::string_literal, Begin, Pos);
    }
    ++Pos;
  }
  Diag.error(SourceLocation{File, Begin}, "unterminated string literal");
  return makeToken(TokenKind::string_literal, Begin, Pos);
}

Token Lexer::lexChar(uint32_t Begin) {
  // Consume opening '\''
  ++Pos;
  while (!eof()) {
    char c = peek();
    if (c == '\'') {
      if (peek(1) == '\'') { Pos += 2; continue; } // escaped quote
      ++Pos;
      return makeToken(TokenKind::char_literal, Begin, Pos);
    }
    if (c == '\n') {
      Diag.error(SourceLocation{File, Begin}, "unterminated character-array literal");
      return makeToken(TokenKind::char_literal, Begin, Pos);
    }
    ++Pos;
  }
  Diag.error(SourceLocation{File, Begin}, "unterminated character-array literal");
  return makeToken(TokenKind::char_literal, Begin, Pos);
}

Token Lexer::lexPunct(uint32_t Begin) {
  char c = peek();
  ++Pos;
  switch (c) {
  case '+': return makeToken(TokenKind::plus, Begin, Pos);
  case '-': return makeToken(TokenKind::minus, Begin, Pos);
  case '*': return makeToken(TokenKind::star, Begin, Pos);
  case '/': return makeToken(TokenKind::slash, Begin, Pos);
  case '\\': return makeToken(TokenKind::backslash, Begin, Pos);
  case '^': return makeToken(TokenKind::caret, Begin, Pos);
  case '(': return makeToken(TokenKind::l_paren, Begin, Pos);
  case ')': return makeToken(TokenKind::r_paren, Begin, Pos);
  case '[': return makeToken(TokenKind::l_square, Begin, Pos);
  case ']': return makeToken(TokenKind::r_square, Begin, Pos);
  case '{': return makeToken(TokenKind::l_brace, Begin, Pos);
  case '}': return makeToken(TokenKind::r_brace, Begin, Pos);
  case ',': return makeToken(TokenKind::comma, Begin, Pos);
  case ';': return makeToken(TokenKind::semi, Begin, Pos);
  case ':': return makeToken(TokenKind::colon, Begin, Pos);
  case '@': return makeToken(TokenKind::at, Begin, Pos);
  case '?': return makeToken(TokenKind::question, Begin, Pos);
  case '=':
    if (eat('=')) return makeToken(TokenKind::eq_eq, Begin, Pos);
    return makeToken(TokenKind::equal, Begin, Pos);
  case '~':
    if (eat('=')) return makeToken(TokenKind::bang_eq, Begin, Pos);
    return makeToken(TokenKind::tilde, Begin, Pos);
  case '<':
    if (eat('=')) return makeToken(TokenKind::less_eq, Begin, Pos);
    return makeToken(TokenKind::less, Begin, Pos);
  case '>':
    if (eat('=')) return makeToken(TokenKind::greater_eq, Begin, Pos);
    return makeToken(TokenKind::greater, Begin, Pos);
  case '&':
    if (eat('&')) return makeToken(TokenKind::amp_amp, Begin, Pos);
    return makeToken(TokenKind::amp, Begin, Pos);
  case '|':
    if (eat('|')) return makeToken(TokenKind::pipe_pipe, Begin, Pos);
    return makeToken(TokenKind::pipe, Begin, Pos);
  case '.':
    if (eat('*')) return makeToken(TokenKind::dot_star, Begin, Pos);
    if (eat('/')) return makeToken(TokenKind::dot_slash, Begin, Pos);
    if (eat('\\')) return makeToken(TokenKind::dot_backslash, Begin, Pos);
    if (eat('^')) return makeToken(TokenKind::dot_caret, Begin, Pos);
    if (eat('\'')) return makeToken(TokenKind::dot_apostrophe, Begin, Pos);
    return makeToken(TokenKind::dot, Begin, Pos);
  case '\'':
    // Should have been handled by the transpose/string disambiguator above.
    return makeToken(TokenKind::apostrophe, Begin, Pos);
  default:
    Diag.error(SourceLocation{File, Begin},
               std::string("unexpected character '") + c + "'");
    return makeToken(TokenKind::unknown, Begin, Pos);
  }
}

Token Lexer::next() {
  PendingLeadingSpace = false;
  skipHorizontalSpaceAndLineContinuations();

  if (eof()) {
    Token T = makeToken(TokenKind::eof, Pos, Pos);
    PrevKind = T.Kind;
    return T;
  }

  uint32_t Begin = Pos;
  char c = peek();

  // Newline
  if (c == '\n') {
    ++Pos;
    Token T = makeToken(TokenKind::newline, Begin, Pos);
    PrevKind = T.Kind;
    AtLineStart = true;
    StmtStart = true;
    return T;
  }

  // Identifier / keyword
  if (isIdentStart(c)) {
    Token T = lexIdentifierOrKeyword(Begin);
    PrevKind = T.Kind;
    AtLineStart = false;
    StmtStart = false;
    return T;
  }

  // Number (may start with '.digit')
  if (std::isdigit((unsigned char)c) ||
      (c == '.' && std::isdigit((unsigned char)peek(1)))) {
    Token T = lexNumber(Begin);
    PrevKind = T.Kind;
    AtLineStart = false;
    StmtStart = false;
    return T;
  }

  // Double-quoted string
  if (c == '"') {
    Token T = lexString(Begin);
    PrevKind = T.Kind;
    AtLineStart = false;
    StmtStart = false;
    return T;
  }

  // Apostrophe: transpose vs char literal
  if (c == '\'') {
    if (prevAllowsTranspose()) {
      ++Pos;
      Token T = makeToken(TokenKind::apostrophe, Begin, Pos);
      PrevKind = T.Kind;
      AtLineStart = false;
      StmtStart = false;
      return T;
    }
    Token T = lexChar(Begin);
    PrevKind = T.Kind;
    AtLineStart = false;
    StmtStart = false;
    return T;
  }

  Token T = lexPunct(Begin);
  PrevKind = T.Kind;
  AtLineStart = false;
  // statement-terminators reset StmtStart for the *next* token.
  if (T.is(TokenKind::semi) || T.is(TokenKind::comma))
    StmtStart = true;
  else
    StmtStart = false;
  return T;
}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> Tokens;
  while (true) {
    Token T = next();
    if (T.is(TokenKind::eof)) {
      Tokens.push_back(T);
      break;
    }
    Tokens.push_back(T);
  }
  return Tokens;
}

} // namespace matlab
