#include "matlab/StateChart/Interpreter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

namespace matlab::statechart {

namespace {

//===----------------------------------------------------------------------===//
// Tiny tokenizer + recursive-descent evaluator for the chart action
// language. Handles the subset that real charts use:
//   - numeric / boolean / identifier primaries
//   - parenthesised sub-expressions
//   - unary `-` / `!` / `~`
//   - binary `+ - * / == != < > <= >= && || & |`
//   - assignment statements separated by `;` or newline
//   - line comments starting with `%`
// Unknown function calls are evaluated to NaN; the rest of the
// expression machinery copes (NaN propagates through comparisons as
// false, matching MATLAB semantics).
//===----------------------------------------------------------------------===//

enum class Tok {
  End, Num, Ident, Str, LParen, RParen, Comma, Semi,
  Plus, Minus, Star, Slash, Percent,
  Eq, NotEq, Lt, Gt, Le, Ge,
  AndAnd, OrOr, And, Or, Bang, Tilde,
  Assign,
};

struct Token {
  Tok K = Tok::End;
  double Num = 0.0;
  std::string Id;
};

class Lex {
public:
  explicit Lex(const std::string &Src) : Src_(Src) {}

  Token next() {
    skipWs();
    if (Pos_ >= Src_.size()) return {Tok::End};
    char c = Src_[Pos_];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return readNumber();
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') return readIdent();
    // String literals — only ever appear inside `emit('X')` for our
    // purposes, but the lexer accepts them anywhere. The content is
    // stashed in `Token::Id` so the consumer doesn't need a separate
    // payload field.
    if (c == '\'' || c == '"') {
      char Q = c; ++Pos_;
      size_t Start = Pos_;
      while (Pos_ < Src_.size() && Src_[Pos_] != Q) ++Pos_;
      Token T; T.K = Tok::Str;
      T.Id = Src_.substr(Start, Pos_ - Start);
      if (Pos_ < Src_.size()) ++Pos_;
      return T;
    }
    return readOp();
  }

  Token peek() {
    size_t save = Pos_;
    Token T = next();
    Pos_ = save;
    return T;
  }

  bool atEnd() {
    skipWs();
    return Pos_ >= Src_.size();
  }

  bool consumeChar(char c) {
    skipWs();
    if (Pos_ < Src_.size() && Src_[Pos_] == c) { ++Pos_; return true; }
    return false;
  }

  // Source-window accessors used by the duration() raw-text capture.
  // `pos()` is the current cursor (one past the last consumed char);
  // `slice(from, to)` returns the substring exactly as it appeared in
  // source so duration's per-(state, expr) map can key on the
  // user-authored expression text.
  size_t pos() const { return Pos_; }
  const std::string &source() const { return Src_; }
  std::string slice(size_t From, size_t To) const {
    if (To <= From || From >= Src_.size()) return {};
    if (To > Src_.size()) To = Src_.size();
    return Src_.substr(From, To - From);
  }

private:
  const std::string &Src_;
  size_t Pos_ = 0;

  void skipWs() {
    while (Pos_ < Src_.size()) {
      char c = Src_[Pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++Pos_; continue; }
      if (c == '%') {
        while (Pos_ < Src_.size() && Src_[Pos_] != '\n') ++Pos_;
        continue;
      }
      break;
    }
  }

  Token readNumber() {
    size_t start = Pos_;
    while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    if (Pos_ < Src_.size() && Src_[Pos_] == '.') {
      ++Pos_;
      while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    }
    if (Pos_ < Src_.size() && (Src_[Pos_] == 'e' || Src_[Pos_] == 'E')) {
      ++Pos_;
      if (Pos_ < Src_.size() && (Src_[Pos_] == '+' || Src_[Pos_] == '-')) ++Pos_;
      while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    }
    Token T; T.K = Tok::Num;
    try { T.Num = std::stod(Src_.substr(start, Pos_ - start)); } catch (...) { T.Num = 0.0; }
    return T;
  }

  Token readIdent() {
    size_t start = Pos_;
    while (Pos_ < Src_.size() &&
           (std::isalnum(static_cast<unsigned char>(Src_[Pos_])) || Src_[Pos_] == '_'))
      ++Pos_;
    Token T; T.K = Tok::Ident; T.Id = Src_.substr(start, Pos_ - start);
    return T;
  }

  Token readOp() {
    char c = Src_[Pos_];
    auto two = [&](char a, char b, Tok k) -> std::optional<Token> {
      if (Pos_ + 1 < Src_.size() && Src_[Pos_] == a && Src_[Pos_ + 1] == b) {
        Pos_ += 2; Token T; T.K = k; return T;
      }
      return std::nullopt;
    };
    if (auto T = two('=', '=', Tok::Eq))     return *T;
    if (auto T = two('!', '=', Tok::NotEq))  return *T;
    if (auto T = two('~', '=', Tok::NotEq))  return *T;
    if (auto T = two('<', '=', Tok::Le))     return *T;
    if (auto T = two('>', '=', Tok::Ge))     return *T;
    if (auto T = two('&', '&', Tok::AndAnd)) return *T;
    if (auto T = two('|', '|', Tok::OrOr))   return *T;
    ++Pos_;
    Token T;
    switch (c) {
    case '(': T.K = Tok::LParen; break;
    case ')': T.K = Tok::RParen; break;
    case ',': T.K = Tok::Comma; break;
    case ';': T.K = Tok::Semi; break;
    case '+': T.K = Tok::Plus; break;
    case '-': T.K = Tok::Minus; break;
    case '*': T.K = Tok::Star; break;
    case '/': T.K = Tok::Slash; break;
    case '<': T.K = Tok::Lt; break;
    case '>': T.K = Tok::Gt; break;
    case '&': T.K = Tok::And; break;
    case '|': T.K = Tok::Or; break;
    case '!': T.K = Tok::Bang; break;
    case '~': T.K = Tok::Tilde; break;
    case '=': T.K = Tok::Assign; break;
    case '%': T.K = Tok::Percent; break;
    default:  T.K = Tok::End; break;
    }
    return T;
  }
};

// Forward decl — defined after Parser but called from inside it.
bool tryTemporal(const std::string &Name, double N,
                 ChartInterpreter &Interp, double &Out);

class Parser {
public:
  // The parser is intentionally one-shot: a fresh parser instance
  // is built per action/guard source. Charts have small action
  // bodies; building an AST + caching it isn't worth the complexity.
  Parser(const std::string &Src, ChartInterpreter &Interp,
         const Chart &C, std::unordered_map<std::string, double> &Locals,
         std::unordered_set<std::string> &Events)
      : Lex_(Src), Interp_(Interp), Chart_(C), Locals_(Locals), Events_(Events) {
    Cur_ = Lex_.next();
  }

  // Parse + evaluate an expression; consumes one expression. Returns
  // the value; sets `Ok=false` on parse failure.
  double parseExpression(bool &Ok) {
    Ok = true;
    double V = parseOr(Ok);
    return V;
  }

  // Parse + execute a sequence of statements separated by `;` or
  // newline. Two statement forms are recognised:
  //   - `IDENT = expr` — write to the locals map via the callback.
  //   - `expr` — evaluate for side effects only (emit() / etc.). The
  //     returned value is discarded.
  // The two-token lookahead (Lex_.peek of the token after Cur_) lets
  // us distinguish the two without backtracking.
  template <class WriteFn>
  void execStatements(WriteFn write, bool &Ok) {
    Ok = true;
    while (Ok && Cur_.K != Tok::End) {
      while (Cur_.K == Tok::Semi) advance();
      if (Cur_.K == Tok::End) break;
      bool IsAssign = false;
      if (Cur_.K == Tok::Ident) {
        Token Next = Lex_.peek();
        if (Next.K == Tok::Assign) IsAssign = true;
      }
      if (IsAssign) {
        std::string Name = Cur_.Id;
        advance();  // consume IDENT
        advance();  // consume `=`
        double V = parseOr(Ok);
        if (!Ok) return;
        write(Name, V);
      } else {
        (void)parseOr(Ok);
        if (!Ok) return;
      }
      if (Cur_.K == Tok::Semi) advance();
    }
  }

private:
  Lex Lex_;
  Token Cur_;
  ChartInterpreter &Interp_;
  [[maybe_unused]] const Chart &Chart_;
  std::unordered_map<std::string, double> &Locals_;
  std::unordered_set<std::string> &Events_;

  void advance() { Cur_ = Lex_.next(); }

  double parseOr(bool &Ok) {
    double L = parseAnd(Ok);
    while (Ok && (Cur_.K == Tok::OrOr || Cur_.K == Tok::Or)) {
      advance();
      double R = parseAnd(Ok);
      L = ((L != 0.0) || (R != 0.0)) ? 1.0 : 0.0;
    }
    return L;
  }
  double parseAnd(bool &Ok) {
    double L = parseRel(Ok);
    while (Ok && (Cur_.K == Tok::AndAnd || Cur_.K == Tok::And)) {
      advance();
      double R = parseRel(Ok);
      L = ((L != 0.0) && (R != 0.0)) ? 1.0 : 0.0;
    }
    return L;
  }
  double parseRel(bool &Ok) {
    double L = parseAdd(Ok);
    if (!Ok) return L;
    Tok K = Cur_.K;
    if (K == Tok::Eq || K == Tok::NotEq || K == Tok::Lt ||
        K == Tok::Gt || K == Tok::Le    || K == Tok::Ge) {
      advance();
      double R = parseAdd(Ok);
      switch (K) {
      case Tok::Eq:    return L == R ? 1.0 : 0.0;
      case Tok::NotEq: return L != R ? 1.0 : 0.0;
      case Tok::Lt:    return L <  R ? 1.0 : 0.0;
      case Tok::Gt:    return L >  R ? 1.0 : 0.0;
      case Tok::Le:    return L <= R ? 1.0 : 0.0;
      case Tok::Ge:    return L >= R ? 1.0 : 0.0;
      default: break;
      }
    }
    return L;
  }
  double parseAdd(bool &Ok) {
    double L = parseMul(Ok);
    while (Ok && (Cur_.K == Tok::Plus || Cur_.K == Tok::Minus)) {
      Tok K = Cur_.K; advance();
      double R = parseMul(Ok);
      L = (K == Tok::Plus) ? L + R : L - R;
    }
    return L;
  }
  double parseMul(bool &Ok) {
    double L = parseUnary(Ok);
    while (Ok && (Cur_.K == Tok::Star || Cur_.K == Tok::Slash)) {
      Tok K = Cur_.K; advance();
      double R = parseUnary(Ok);
      L = (K == Tok::Star) ? L * R : (R == 0.0 ? std::nan("") : L / R);
    }
    return L;
  }
  double parseUnary(bool &Ok) {
    if (Cur_.K == Tok::Minus) { advance(); return -parseUnary(Ok); }
    if (Cur_.K == Tok::Plus)  { advance(); return  parseUnary(Ok); }
    if (Cur_.K == Tok::Bang || Cur_.K == Tok::Tilde) {
      advance();
      double V = parseUnary(Ok);
      return (V == 0.0) ? 1.0 : 0.0;
    }
    return parsePrimary(Ok);
  }
  double parsePrimary(bool &Ok) {
    if (Cur_.K == Tok::Num) { double v = Cur_.Num; advance(); return v; }
    // A string literal in any non-`emit` / non-`in` position has no
    // scalar interpretation — fall through with 0 so the caller's
    // expression chain stays well-formed.
    if (Cur_.K == Tok::Str) { advance(); return 0.0; }
    if (Cur_.K == Tok::LParen) {
      advance();
      double v = parseOr(Ok);
      if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
      advance();
      return v;
    }
    if (Cur_.K == Tok::Ident) {
      std::string Name = Cur_.Id;
      advance();
      if (Cur_.K == Tok::LParen) {
        // function call — collect args, dispatch builtins.
        // Snapshot the lexer cursor right after `(` (before any inner
        // tokens are consumed) so duration() can later carve out the
        // raw expression source text by slicing [start, end).
        size_t LParenPast = Lex_.pos();
        advance();
        // `duration(EXPR)` — capture the raw inner-expression text
        // by slicing the lexer's source between the open paren and
        // the matching close paren, then look up the (state, expr)
        // slot. The bool value of EXPR drives the slot transitions;
        // the duration in ticks is returned to the caller.
        if (Name == "duration") {
          double Cond = parseOr(Ok);
          if (!Ok) return 0.0;
          if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
          // Lex_.pos() at this point is one past the `)` that the
          // last advance() consumed when Cur_ became RParen.
          size_t InnerEnd = Lex_.pos();
          if (InnerEnd > 0) --InnerEnd;  // step back over `)`
          std::string Expr = Lex_.slice(LParenPast, InnerEnd);
          while (!Expr.empty() &&
                 std::isspace((unsigned char)Expr.front()))
            Expr.erase(Expr.begin());
          while (!Expr.empty() &&
                 std::isspace((unsigned char)Expr.back()))
            Expr.pop_back();
          advance();  // consume `)`
          return (double)Interp_.durationOf(
              Interp_.actionOwner(), Expr, Cond != 0.0);
        }
        // emit('X') / emit("X") / emit(X) — broadcast the named event
        // into the interpreter's pending-events set. Returns 0 as a
        // scalar so it composes with assignment statements (rare but
        // legal Stateflow shorthand). Statement-context users
        // typically write `emit(X);`.
        if (Name == "emit" &&
            (Cur_.K == Tok::Ident || Cur_.K == Tok::Str)) {
          std::string EvName = Cur_.Id;
          advance();
          if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
          advance();
          Interp_.emit(EvName);
          return 0.0;
        }
        std::vector<double> Args;
        if (Cur_.K != Tok::RParen) {
          while (true) {
            // Special-case `in(stateId)`: a single identifier or
            // string literal naming the state to query.
            if (Name == "in" &&
                (Cur_.K == Tok::Ident || Cur_.K == Tok::Str)) {
              std::string StateId = Cur_.Id;
              advance();
              if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
              advance();
              return Interp_.isActive(StateId) ? 1.0 : 0.0;
            }
            // `temporalCount(eventName)`: counter-style temporal —
            // returns the count of broadcasts of `eventName` since
            // the owner state was last entered. The chart's
            // super-step processor increments these counters per
            // active state when an event fires (see
            // ChartInterpreter::superStep).
            if (Name == "temporalCount" &&
                (Cur_.K == Tok::Ident || Cur_.K == Tok::Str)) {
              std::string Ev = Cur_.Id;
              advance();
              if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
              advance();
              return (double)Interp_.tempCountOf(
                  Interp_.actionOwner(), Ev);
            }
            double v = parseOr(Ok);
            if (!Ok) return 0.0;
            Args.push_back(v);
            if (Cur_.K == Tok::Comma) { advance(); continue; }
            break;
          }
        }
        if (Cur_.K != Tok::RParen) { Ok = false; return 0.0; }
        advance();
        // Temporal operators — handled BEFORE generic builtin dispatch
        // because they consult the interpreter's tick counter and the
        // current action's owning state id rather than just their
        // numeric arguments.
        double TempOut = 0.0;
        if (Args.size() >= 1 && tryTemporal(Name, Args[0], Interp_, TempOut))
          return TempOut;
        return dispatchBuiltin(Name, Args);
      }
      // Bare identifier — events first (charts often guard on event
      // names directly), then locals, then named constants.
      if (Events_.count(Name)) {
        // Event references in a guard read whether the event is
        // currently broadcast.
        return Interp_.activeStates().empty() ? 0.0 :
               (Interp_.isActive("__never__") ? 1.0 : 0.0);
        // The above is misleading on purpose — guard event refs come
        // from the parsed `Label.Event` slot, NOT from the guard
        // body. Inside a guard body, a bare event name should never
        // resolve through the interpreter's event flag — the caller
        // has already done that. We return 0 so a bare event ref in
        // an expression yields false, which is the safe interpretation.
      }
      auto It = Locals_.find(Name);
      if (It != Locals_.end()) return It->second;
      if (Name == "true")  return 1.0;
      if (Name == "false") return 0.0;
      if (Name == "pi")    return 3.14159265358979323846;
      if (Name == "Inf")   return std::numeric_limits<double>::infinity();
      if (Name == "NaN")   return std::nan("");
      // Unknown identifier — treat as 0 with no diagnostic. Charts
      // that reference unknown symbols will surface in matlabc when
      // the lowered MATLAB compiles; the interpreter just runs.
      return 0.0;
    }
    Ok = false;
    return 0.0;
  }

  static double dispatchBuiltin(const std::string &Name,
                                const std::vector<double> &A) {
    auto a = [&](size_t i) { return i < A.size() ? A[i] : 0.0; };
    if (Name == "abs"   && A.size() >= 1) return std::fabs(a(0));
    if (Name == "min"   && A.size() >= 2) return std::min(a(0), a(1));
    if (Name == "max"   && A.size() >= 2) return std::max(a(0), a(1));
    if (Name == "floor" && A.size() >= 1) return std::floor(a(0));
    if (Name == "ceil"  && A.size() >= 1) return std::ceil(a(0));
    if (Name == "round" && A.size() >= 1) return std::round(a(0));
    if (Name == "mod"   && A.size() >= 2) return a(1) == 0 ? 0.0 : a(0) - std::floor(a(0) / a(1)) * a(1);
    if (Name == "sqrt"  && A.size() >= 1) return std::sqrt(a(0));
    if (Name == "sin"   && A.size() >= 1) return std::sin(a(0));
    if (Name == "cos"   && A.size() >= 1) return std::cos(a(0));
    if (Name == "exp"   && A.size() >= 1) return std::exp(a(0));
    if (Name == "log"   && A.size() >= 1) return std::log(a(0));
    return std::nan("");
  }
};

// Free helper used by the parser's primary path to evaluate the
// temporal builtins. Returns true if the call site is a temporal
// operator (and writes the value into `Out`); false otherwise. The
// `unit` token is consumed (and ignored) for tick/sec parity.
bool tryTemporal(const std::string &Name, double N,
                 ChartInterpreter &Interp, double &Out) {
  if (Interp.actionOwner().empty()) return false;
  int Delta = Interp.tickCount() - Interp.entryTimeOf(Interp.actionOwner());
  if (Name == "after")  { Out = Delta >= (int)N ? 1.0 : 0.0; return true; }
  if (Name == "before") { Out = Delta <  (int)N ? 1.0 : 0.0; return true; }
  if (Name == "at")     { Out = Delta == (int)N ? 1.0 : 0.0; return true; }
  if (Name == "every") {
    int Per = (int)N;
    if (Per <= 0) { Out = 0.0; return true; }
    Out = (Delta > 0 && (Delta % Per) == 0) ? 1.0 : 0.0;
    return true;
  }
  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// ChartInterpreter — implementation
//===----------------------------------------------------------------------===//

ChartInterpreter::ChartInterpreter(const Chart &C)
    : MaxIterations(C.MaxIterations), C_(C) {
  // Seed locals from the symbol table's initial values.
  for (auto &S : C_.Symbols.Data) {
    double V = 0.0;
    if (!S.Initial.empty()) {
      try { V = std::stod(S.Initial); }
      catch (...) { V = 0.0; }
    }
    Locals_[S.Name] = V;
  }
  // Inputs / outputs still need slots even when omitted from the data
  // table — driver scripts set them by name.
  for (auto &N : C_.Sig.Inputs)  if (!Locals_.count(N)) Locals_[N] = 0.0;
  for (auto &N : C_.Sig.Outputs) if (!Locals_.count(N)) Locals_[N] = 0.0;
}

void ChartInterpreter::emit(const std::string &EventName) {
  Events_.insert(EventName);
}

int ChartInterpreter::durationOf(const std::string &State,
                                 const std::string &Expr, bool CondHolds) {
  auto &Slot = Durations_[{State, Expr}];
  if (CondHolds) {
    if (Slot.first == 0) {
      Slot.first  = 1;
      Slot.second = TickCount_;
      return 0;
    }
    return TickCount_ - Slot.second;
  }
  Slot.first  = 0;
  Slot.second = 0;
  return 0;
}

void ChartInterpreter::setLocal(const std::string &Name, double V) {
  Locals_[Name] = V;
}

std::optional<double>
ChartInterpreter::getLocal(const std::string &Name) const {
  auto It = Locals_.find(Name);
  if (It == Locals_.end()) return std::nullopt;
  return It->second;
}

std::vector<std::pair<std::string, double>>
ChartInterpreter::allLocals() const {
  std::vector<std::pair<std::string, double>> Out;
  Out.reserve(Locals_.size());
  for (auto &P : Locals_) Out.emplace_back(P.first, P.second);
  std::sort(Out.begin(), Out.end(),
            [](const auto &A, const auto &B) { return A.first < B.first; });
  return Out;
}

bool ChartInterpreter::isActive(const std::string &StateId) const {
  // Walk the parent chain: a state is active iff every ancestor's
  // OR-region slot resolves to one of its substates that's on the
  // path to StateId. For AND parents the child is always considered
  // active provided the AND parent itself is.
  const ChartState *S = C_.findState(StateId);
  if (!S) return false;
  std::string Cur = StateId;
  while (true) {
    if (Cur.empty()) {
      // Reached chart root — final check: chart_root slot must
      // contain the topmost ancestor we just verified.
      return true;
    }
    const ChartState *CurS = C_.findState(Cur);
    if (!CurS) return false;
    const std::string Parent = CurS->ParentId;
    const ChartState *ParentS =
        Parent.empty() ? nullptr : C_.findState(Parent);
    if (ParentS && ParentS->Decomp == Decomposition::And) {
      // AND parent: the child is active iff the parent is. Step up.
      Cur = Parent;
      continue;
    }
    // OR parent (or chart root): slot must equal Cur.
    std::string Key = Parent.empty() ? "chart_root" : Parent;
    auto It = Regions_.find(Key);
    if (It == Regions_.end() || It->second != Cur) return false;
    Cur = Parent;
  }
}

std::vector<std::string> ChartInterpreter::activeStates() const {
  std::vector<std::string> Out;
  // Walk from chart root following region slots / AND children.
  std::function<void(const std::string &)> Visit =
      [&](const std::string &Sid) {
        if (Sid.empty()) return;
        Out.push_back(Sid);
        const ChartState *S = C_.findState(Sid);
        if (!S) return;
        if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
          auto It = Regions_.find(Sid);
          if (It != Regions_.end()) Visit(It->second);
        } else if (S->Decomp == Decomposition::And) {
          // Children sorted by exec order for deterministic output.
          std::vector<std::string> Kids = S->ChildStateIds;
          std::stable_sort(Kids.begin(), Kids.end(),
              [&](const std::string &A, const std::string &B) {
                int Ai = 0, Bi = 0;
                if (auto *Sa = C_.findState(A))
                  Ai = Sa->ExecutionOrder.value_or(0);
                if (auto *Sb = C_.findState(B))
                  Bi = Sb->ExecutionOrder.value_or(0);
                return Ai < Bi;
              });
          for (auto &K : Kids) Visit(K);
        }
      };
  auto It = Regions_.find("chart_root");
  if (It != Regions_.end()) Visit(It->second);
  return Out;
}

void ChartInterpreter::setStateEnterBreakpoints(const std::vector<std::string> &Ids) {
  StateEnterBP_.clear(); for (auto &I : Ids) StateEnterBP_.insert(I);
}
void ChartInterpreter::setStateExitBreakpoints(const std::vector<std::string> &Ids) {
  StateExitBP_.clear(); for (auto &I : Ids) StateExitBP_.insert(I);
}
void ChartInterpreter::setTransitionBreakpoints(const std::vector<std::string> &Ids) {
  TransitionBP_.clear(); for (auto &I : Ids) TransitionBP_.insert(I);
}
void ChartInterpreter::setSymbolBreakpoints(const std::vector<std::string> &Names) {
  SymbolBP_.clear(); for (auto &N : Names) SymbolBP_.insert(N);
}

std::string ChartInterpreter::initialSubstateOf(const ChartState &S) const {
  // Prefer history junction (Tier 4 partial — only the
  // `params.hasHistory` flag is observed; sibling
  // `junction_history` nodes are recognised but not yet wired).
  if (S.HasHistory) {
    auto It = History_.find(S.Id);
    if (It != History_.end()) return It->second;
  }
  for (auto &Cid : S.ChildStateIds) {
    const ChartState *Cs = C_.findState(Cid);
    if (Cs && Cs->IsInitial) return Cid;
  }
  // Fall back to a `junction_default` sibling's outgoing edge.
  for (auto &P : C_.Junctions) {
    const ChartJunction &J = P.second;
    if (J.Kind != JunctionKind::Default) continue;
    if (J.ParentId != S.Id) continue;
    for (auto &T : C_.Transitions)
      if (T.SourceId == J.Id) return T.DestId;
  }
  return S.ChildStateIds.empty() ? std::string() : S.ChildStateIds.front();
}

std::vector<const Transition *>
ChartInterpreter::outgoingFrom(const std::string &Id) const {
  std::vector<const Transition *> Out;
  for (auto &T : C_.Transitions)
    if (T.SourceId == Id) Out.push_back(&T);
  std::stable_sort(Out.begin(), Out.end(),
      [](const Transition *A, const Transition *B) {
        return A->Priority < B->Priority;
      });
  return Out;
}

double ChartInterpreter::evalExpression(const std::string &Src) {
  Parser P(Src, *this, C_, Locals_, Events_);
  bool Ok = true;
  return P.parseExpression(Ok);
}

void ChartInterpreter::execAction(const std::string &Src) {
  if (Src.empty()) return;
  Parser P(Src, *this, C_, Locals_, Events_);
  bool Ok = true;
  P.execStatements(
      [&](const std::string &Name, double Value) {
        // Symbol-change watchpoint — fires AFTER the value lands so
        // the IDE can read the new state via `getLocals`.
        bool Changed = Locals_.find(Name) == Locals_.end() ||
                       Locals_[Name] != Value;
        Locals_[Name] = Value;
        if (Changed && SymbolBP_.count(Name) && ActionTrace_) {
          ChartTraceEvent BP;
          BP.K = ChartTraceEvent::Kind::Breakpoint;
          BP.Id = Name;
          BP.BreakpointReason = "symbolChange";
          ActionTrace_->push_back(std::move(BP));
          BreakpointPending_ = true;
        }
      },
      Ok);
}

bool ChartInterpreter::evalGuard(const Transition &T) {
  // Event check first — a non-empty event field gates the transition.
  if (!T.Label.Event.empty()) {
    bool HasSym = false;
    for (auto &S : C_.Symbols.Events)
      if (S.Name == T.Label.Event) { HasSym = true; break; }
    if (HasSym && !Events_.count(T.Label.Event)) return false;
  }
  if (T.Label.Guard.empty()) return true;
  // Owner of a transition's guard is the source state so temporal
  // operators (e.g. `[after(30, sec)]`) compute time-since-entry of
  // that state.
  ActionOwner_ = T.SourceId;
  double V = evalExpression(T.Label.Guard);
  return V != 0.0;
}

void ChartInterpreter::enterState(const std::string &Id,
                                  std::vector<ChartTraceEvent> &Out) {
  const ChartState *S = C_.findState(Id);
  if (!S) return;
  // Mark active slot on OR / chart-root parents.
  const ChartState *Parent =
      S->ParentId.empty() ? nullptr : C_.findState(S->ParentId);
  bool ParentHasSlot =
      S->ParentId.empty() ||
      (Parent && Parent->Decomp == Decomposition::Or);
  if (ParentHasSlot) {
    std::string Key = S->ParentId.empty() ? "chart_root" : S->ParentId;
    Regions_[Key] = Id;
  }
  // Stamp entry time for temporal operators.
  EntryTimes_[Id] = TickCount_;
  // Clear counter-style temporal slots so a fresh entry observes
  // temporalCount==0 + duration==0. Iterate and erase any (state, *)
  // pair whose first element is `Id`.
  for (auto It = TempCounts_.begin(); It != TempCounts_.end(); ) {
    if (It->first.first == Id) It = TempCounts_.erase(It);
    else ++It;
  }
  for (auto It = Durations_.begin(); It != Durations_.end(); ) {
    if (It->first.first == Id) It = Durations_.erase(It);
    else ++It;
  }
  // Entry action — owner is this state so temporal operators inside
  // resolve against EntryTimes_[Id].
  ActionOwner_ = Id;
  execAction(S->Entry.Source);
  // Trace + breakpoint.
  {
    ChartTraceEvent E; E.K = ChartTraceEvent::Kind::StateEnter; E.Id = Id;
    Out.push_back(std::move(E));
  }
  if (StateEnterBP_.count(Id)) {
    ChartTraceEvent BP;
    BP.K = ChartTraceEvent::Kind::Breakpoint;
    BP.Id = Id;
    BP.BreakpointReason = "stateEnter";
    Out.push_back(std::move(BP));
    BreakpointPending_ = true;
  }
  // Recurse.
  if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
    std::string Init = initialSubstateOf(*S);
    if (!Init.empty()) enterState(Init, Out);
  } else if (S->Decomp == Decomposition::And) {
    std::vector<std::string> Kids = S->ChildStateIds;
    std::stable_sort(Kids.begin(), Kids.end(),
        [&](const std::string &A, const std::string &B) {
          int Ai = 0, Bi = 0;
          if (auto *Sa = C_.findState(A))
            Ai = Sa->ExecutionOrder.value_or(0);
          if (auto *Sb = C_.findState(B))
            Bi = Sb->ExecutionOrder.value_or(0);
          return Ai < Bi;
        });
    for (auto &K : Kids) enterState(K, Out);
  }
}

void ChartInterpreter::exitState(const std::string &Id,
                                 std::vector<ChartTraceEvent> &Out) {
  const ChartState *S = C_.findState(Id);
  if (!S) return;
  // Recurse first — deepest substates exit before their parent.
  if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
    auto It = Regions_.find(Id);
    if (It != Regions_.end() && !It->second.empty()) {
      // Save history so a future hasHistory entry can restore it.
      History_[Id] = It->second;
      exitState(It->second, Out);
      Regions_[Id] = "";
    }
  } else if (S->Decomp == Decomposition::And) {
    std::vector<std::string> Kids = S->ChildStateIds;
    std::stable_sort(Kids.begin(), Kids.end(),
        [&](const std::string &A, const std::string &B) {
          int Ai = 0, Bi = 0;
          if (auto *Sa = C_.findState(A))
            Ai = Sa->ExecutionOrder.value_or(0);
          if (auto *Sb = C_.findState(B))
            Bi = Sb->ExecutionOrder.value_or(0);
          return Bi < Ai;  // reverse exec order on exit
        });
    for (auto &K : Kids) exitState(K, Out);
  }
  // Exit action — owner is the exiting state.
  ActionOwner_ = Id;
  execAction(S->Exit.Source);
  {
    ChartTraceEvent E; E.K = ChartTraceEvent::Kind::StateExit; E.Id = Id;
    Out.push_back(std::move(E));
  }
  if (StateExitBP_.count(Id)) {
    ChartTraceEvent BP;
    BP.K = ChartTraceEvent::Kind::Breakpoint;
    BP.Id = Id;
    BP.BreakpointReason = "stateExit";
    Out.push_back(std::move(BP));
    BreakpointPending_ = true;
  }
}

bool ChartInterpreter::walkPath(const std::string &NodeId,
                                std::vector<const Transition *> &Segs,
                                std::string &FinalState) {
  if (C_.findState(NodeId)) { FinalState = NodeId; return true; }
  const ChartJunction *J = C_.findJunction(NodeId);
  if (!J) return false;
  // history junction: redirect to the parent state's last-active
  // substate (or the parent's initial if no history).
  if (J->Kind == JunctionKind::History) {
    auto HIt = History_.find(J->ParentId);
    std::string Target = (HIt != History_.end()) ? HIt->second
                                                  : std::string();
    if (Target.empty()) {
      const ChartState *Parent = C_.findState(J->ParentId);
      if (Parent) Target = initialSubstateOf(*Parent);
    }
    if (Target.empty()) return false;
    return walkPath(Target, Segs, FinalState);
  }
  // Connective / entry / exit / default: try each outgoing in
  // priority order; the first whose guard passes AND whose downstream
  // reaches a state wins.
  for (auto *Out : outgoingFrom(NodeId)) {
    if (!evalGuard(*Out)) continue;
    // Cond action of every committed segment is evaluated even before
    // we know whether the chain commits — Stateflow §2-29. To keep
    // the side-effect model simple we run cond/trans on commit only
    // (documented Tier-4 deviation).
    Segs.push_back(Out);
    std::string Sub;
    if (walkPath(Out->DestId, Segs, Sub)) {
      FinalState = Sub;
      return true;
    }
    Segs.pop_back();
  }
  return false;
}

std::optional<ChartInterpreter::ResolvedPath>
ChartInterpreter::resolvePath(const Transition &T) {
  ResolvedPath R;
  R.Segments.push_back(&T);
  std::string Final;
  if (!walkPath(T.DestId, R.Segments, Final)) return std::nullopt;
  R.FinalStateId = Final;
  return R;
}

void ChartInterpreter::fireTransition(const Transition &T,
                                      std::vector<ChartTraceEvent> &Out) {
  // Resolve the full path — for a sibling transition this just
  // returns {T, T.DestId}; for a junction chain it threads through
  // every connective / entry / exit hop with a passing guard.
  auto Path = resolvePath(T);
  if (!Path) return;  // shouldn't happen — caller already verified
  // Inner transitions skip the exit / entry chain entirely
  // (Stateflow §1-45). The cond + trans actions still run.
  bool IsInner = (T.Kind == TransitionKind::Inner);
  // Cond actions accumulate first.
  for (auto *Seg : Path->Segments) execAction(Seg->Label.CondAction);
  // Super-transition support: walk the source up to the LCA with
  // the destination, exiting each level; then walk the LCA down to
  // the destination, entering each level. For sibling transitions
  // (src + dst share immediate parent) the LCA is that parent and
  // the chain reduces to a single exit + entry.
  auto ancestors = [&](const std::string &Id) {
    std::vector<std::string> A;
    std::string Cur = Id;
    while (!Cur.empty()) {
      A.push_back(Cur);
      const ChartState *S = C_.findState(Cur);
      const ChartJunction *J = C_.findJunction(Cur);
      Cur = S ? S->ParentId : (J ? J->ParentId : std::string());
    }
    return A;
  };
  auto lca = [&](const std::string &A,
                 const std::string &B) -> std::string {
    auto Aa = ancestors(A);
    std::unordered_set<std::string> Bs;
    for (auto &S : ancestors(B)) Bs.insert(S);
    for (auto &S : Aa) if (Bs.count(S)) return S;
    return "";
  };
  if (!IsInner && C_.findState(T.SourceId)) {
    std::string LCA = lca(T.SourceId, Path->FinalStateId);
    // Walk src up to (but not including) LCA. exitState already
    // recurses into each level's active substates, so we only need
    // to call it on each state in the chain.
    std::string Cur = T.SourceId;
    while (!Cur.empty() && Cur != LCA) {
      const ChartState *S = C_.findState(Cur);
      if (!S) break;
      // Only call exitState on the deepest level — its recursion
      // handles the substate chain. For higher levels, just run
      // their exit action + clear their slot directly.
      if (Cur == T.SourceId) {
        exitState(Cur, Out);
      } else {
        execAction(S->Exit.Source);
        ChartTraceEvent E; E.K = ChartTraceEvent::Kind::StateExit;
        E.Id = Cur; Out.push_back(std::move(E));
        if (StateExitBP_.count(Cur)) {
          ChartTraceEvent BP;
          BP.K = ChartTraceEvent::Kind::Breakpoint;
          BP.Id = Cur;
          BP.BreakpointReason = "stateExit";
          Out.push_back(std::move(BP));
          BreakpointPending_ = true;
        }
        // Save history if the now-exiting state has children with
        // an OR slot — preserve last-active for re-entry.
        auto RIt = Regions_.find(Cur);
        if (RIt != Regions_.end() && !RIt->second.empty()) {
          History_[Cur] = RIt->second;
          Regions_[Cur] = "";
        }
      }
      Cur = S->ParentId;
    }
  }
  // Trans actions in segment order.
  for (auto *Seg : Path->Segments) execAction(Seg->Label.TransAction);
  // Entry chain on the final state — skip for inner. Walk LCA down
  // to final, entering each level so every OR ancestor's region
  // slot gets set.
  if (!IsInner && C_.findState(Path->FinalStateId)) {
    std::string LCA = lca(T.SourceId, Path->FinalStateId);
    // Build the chain from LCA's child down to FinalState.
    std::vector<std::string> Chain;
    std::string Cur = Path->FinalStateId;
    while (!Cur.empty() && Cur != LCA) {
      Chain.push_back(Cur);
      const ChartState *S = C_.findState(Cur);
      if (!S) break;
      Cur = S->ParentId;
    }
    std::reverse(Chain.begin(), Chain.end());
    for (size_t I = 0; I + 1 < Chain.size(); ++I) {
      // Enter each intermediate level shallow (set slot + run entry
      // action). Skip descending into substates because the chain
      // already names the substate that should be active.
      const ChartState *S = C_.findState(Chain[I]);
      if (!S) continue;
      const ChartState *Parent =
          S->ParentId.empty() ? nullptr : C_.findState(S->ParentId);
      bool ParentHasSlot =
          S->ParentId.empty() ||
          (Parent && Parent->Decomp == Decomposition::Or);
      if (ParentHasSlot)
        Regions_[S->ParentId.empty() ? "chart_root" : S->ParentId] =
            Chain[I];
      execAction(S->Entry.Source);
      ChartTraceEvent E; E.K = ChartTraceEvent::Kind::StateEnter;
      E.Id = Chain[I]; Out.push_back(std::move(E));
      if (StateEnterBP_.count(Chain[I])) {
        ChartTraceEvent BP;
        BP.K = ChartTraceEvent::Kind::Breakpoint;
        BP.Id = Chain[I];
        BP.BreakpointReason = "stateEnter";
        Out.push_back(std::move(BP));
        BreakpointPending_ = true;
      }
    }
    // Deepest level — full enter (descends into substates).
    enterState(Path->FinalStateId, Out);
  }
  // One trace + BP entry per segment so the IDE sees the full chain.
  for (auto *Seg : Path->Segments) {
    ChartTraceEvent E;
    E.K = ChartTraceEvent::Kind::TransitionFired;
    E.Id = Seg->Id;
    E.Src = Seg->SourceId;
    E.Dst = Seg->DestId;
    E.EventName = Seg->Label.Event;
    Out.push_back(std::move(E));
    if (TransitionBP_.count(Seg->Id)) {
      ChartTraceEvent BP;
      BP.K = ChartTraceEvent::Kind::Breakpoint;
      BP.Id = Seg->Id;
      BP.BreakpointReason = "transition";
      Out.push_back(std::move(BP));
      BreakpointPending_ = true;
    }
  }
}

bool ChartInterpreter::stepActiveSubstate(const std::string &SubstateId,
                                          std::vector<ChartTraceEvent> &Out,
                                          bool StopOnFirstTransition) {
  if (SubstateId.empty()) return false;
  const ChartState *S = C_.findState(SubstateId);
  if (!S) return false;

  // First try outgoing transitions of this state in priority order.
  // A transition only fires if the guard passes AND a junction chain
  // (if any) can reach a final state — otherwise we backtrack and
  // try the next candidate.
  for (auto *T : outgoingFrom(SubstateId)) {
    if (!evalGuard(*T)) continue;
    if (!resolvePath(*T)) continue;
    fireTransition(*T, Out);
    return true;
  }

  // No transition fired — run on-event handlers + during action.
  ActionOwner_ = SubstateId;
  for (auto &OE : S->OnEvent) {
    if (Events_.count(OE.first)) execAction(OE.second.Source);
  }
  execAction(S->During.Source);

  // Recurse — OR descends into substate; AND walks children.
  if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
    auto It = Regions_.find(SubstateId);
    if (It != Regions_.end())
      return stepActiveSubstate(It->second, Out, StopOnFirstTransition);
  } else if (S->Decomp == Decomposition::And) {
    bool Fired = false;
    std::vector<std::string> Kids = S->ChildStateIds;
    std::stable_sort(Kids.begin(), Kids.end(),
        [&](const std::string &A, const std::string &B) {
          int Ai = 0, Bi = 0;
          if (auto *Sa = C_.findState(A))
            Ai = Sa->ExecutionOrder.value_or(0);
          if (auto *Sb = C_.findState(B))
            Bi = Sb->ExecutionOrder.value_or(0);
          return Ai < Bi;
        });
    for (auto &K : Kids) {
      if (BreakpointPending_) break;
      if (stepActiveSubstate(K, Out, StopOnFirstTransition)) {
        Fired = true;
        if (StopOnFirstTransition) break;
      }
    }
    return Fired;
  }
  return false;
}

bool ChartInterpreter::stepChartRoot(std::vector<ChartTraceEvent> &Out,
                                     bool StopOnFirstTransition) {
  auto It = Regions_.find("chart_root");
  if (It == Regions_.end() || It->second.empty()) return false;
  return stepActiveSubstate(It->second, Out, StopOnFirstTransition);
}

void ChartInterpreter::initialEnter(std::vector<ChartTraceEvent> &Out) {
  // Walk root substates: prefer isInitial, else default junction, else first.
  std::string Init;
  for (auto &Id : C_.RootStateIds) {
    const ChartState *S = C_.findState(Id);
    if (S && S->IsInitial) { Init = Id; break; }
  }
  if (Init.empty()) {
    for (auto &P : C_.Junctions) {
      const ChartJunction &J = P.second;
      if (J.Kind != JunctionKind::Default || !J.ParentId.empty()) continue;
      for (auto &T : C_.Transitions)
        if (T.SourceId == J.Id) { Init = T.DestId; break; }
      if (!Init.empty()) break;
    }
  }
  if (Init.empty() && !C_.RootStateIds.empty()) Init = C_.RootStateIds.front();
  if (!Init.empty()) enterState(Init, Out);
}

std::vector<ChartTraceEvent> ChartInterpreter::initialize() {
  std::vector<ChartTraceEvent> Out;
  if (Initialized_) return Out;
  Initialized_ = true;
  ActionTrace_ = &Out;
  ChartTraceEvent Begin; Begin.K = ChartTraceEvent::Kind::SuperStepBegin;
  Begin.Iteration = 0; Out.push_back(std::move(Begin));
  initialEnter(Out);
  ChartTraceEvent End; End.K = ChartTraceEvent::Kind::SuperStepEnd;
  End.Iteration = 0; End.Quiescent = true; Out.push_back(std::move(End));
  ActionTrace_ = nullptr;
  return Out;
}

std::vector<ChartTraceEvent> ChartInterpreter::superStep() {
  std::vector<ChartTraceEvent> Out;
  if (!Initialized_) {
    auto Init = initialize();
    Out.insert(Out.end(), Init.begin(), Init.end());
  }
  ++TickCount_;
  ActionTrace_ = &Out;
  // Mirror each broadcast event so the trace records the user-driven
  // input that powered this super-step.
  for (auto &E : Events_) {
    ChartTraceEvent EB;
    EB.K = ChartTraceEvent::Kind::EventBroadcast;
    EB.Id = E; Out.push_back(std::move(EB));
  }
  // Counter-style temporal maintenance: for every event in this
  // super-step's broadcast set, increment temporalCount(event) for
  // every currently-active state. Active states are tracked through
  // Regions_; we walk them via isActive() to catch nested OR/AND
  // parents.
  for (auto &Ev : Events_)
    for (auto &P : C_.States)
      if (isActive(P.first))
        ++TempCounts_[{P.first, Ev}];
  ChartTraceEvent Begin; Begin.K = ChartTraceEvent::Kind::SuperStepBegin;
  Begin.Iteration = 0; Out.push_back(std::move(Begin));
  int Iter = 0;
  bool LastFired = true;
  for (Iter = 0; Iter < MaxIterations && LastFired; ++Iter) {
    if (BreakpointPending_) break;
    LastFired = stepChartRoot(Out, /*StopOnFirstTransition=*/false);
  }
  if (LastFired && Iter >= MaxIterations) {
    ChartTraceEvent MI;
    MI.K = ChartTraceEvent::Kind::MaxIterations;
    MI.Iteration = Iter;
    Out.push_back(std::move(MI));
  }
  bool BrokeByBP = BreakpointPending_;
  ChartTraceEvent End; End.K = ChartTraceEvent::Kind::SuperStepEnd;
  End.Iteration = Iter;
  End.Quiescent = !LastFired && !BrokeByBP;
  Out.push_back(std::move(End));
  Events_.clear();
  BreakpointPending_ = false;
  ActionTrace_ = nullptr;
  return Out;
}

std::vector<ChartTraceEvent> ChartInterpreter::stepTransition() {
  std::vector<ChartTraceEvent> Out;
  if (!Initialized_) {
    auto Init = initialize();
    Out.insert(Out.end(), Init.begin(), Init.end());
    return Out;
  }
  ActionTrace_ = &Out;
  for (auto &E : Events_) {
    ChartTraceEvent EB;
    EB.K = ChartTraceEvent::Kind::EventBroadcast;
    EB.Id = E; Out.push_back(std::move(EB));
  }
  ChartTraceEvent Begin; Begin.K = ChartTraceEvent::Kind::SuperStepBegin;
  Begin.Iteration = 0; Out.push_back(std::move(Begin));
  bool Fired = stepChartRoot(Out, /*StopOnFirstTransition=*/true);
  ChartTraceEvent End; End.K = ChartTraceEvent::Kind::SuperStepEnd;
  End.Iteration = Fired ? 1 : 0;
  End.Quiescent = !Fired;
  Out.push_back(std::move(End));
  // stepTransition leaves events live so a follow-on call can still
  // fire on them. They clear naturally on the next superStep.
  BreakpointPending_ = false;
  ActionTrace_ = nullptr;
  return Out;
}

ChartInterpreter::Snapshot ChartInterpreter::snapshot() const {
  Snapshot S;
  S.Regions = Regions_;
  S.Locals  = Locals_;
  S.History = History_;
  return S;
}

void ChartInterpreter::restore(const Snapshot &S) {
  Regions_ = S.Regions;
  Locals_  = S.Locals;
  History_ = S.History;
  Initialized_ = !Regions_.empty();
  Events_.clear();
  BreakpointPending_ = false;
}

} // namespace matlab::statechart
