#include "matlab/StateChart/Lowering.h"

#include "matlab/Basic/Diagnostic.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace matlab::statechart {

namespace {

//===----------------------------------------------------------------------===//
// JIT-friendly chart lowering.
//
// The output is a MATLAB script that matlabc can compile through
// every downstream lane (`-emit-llvm`, `-emit-c`, `-emit-cpp`, the
// experimental REPL). The chart compiles to a single `<chart>_tick`
// local function plus a top-level demo driver:
//
//   function <out1>, <out2>... = <chart>_tick(<in1>, <in2>, ...,
//                                              <ev_e1>, <ev_e2>, ...)
//     persistent r_<region> ... l_<local> ... t_<state>
//                init_done tick_count
//     if isempty(init_done)
//       <init block>
//       <first-call entry chain>
//       init_done = 1;
//     end
//     tick_count = tick_count + 1;
//     <super-step loop>
//   end
//
// No struct(), no string literals as state values, no nested struct
// field accesses — every chart slot is a flat persistent scalar.
// Regions are integer-typed (1..N for each substate plus 0 for
// "uninitialised"). matlabc's MATLAB→LLVM lane only supports scalar
// arithmetic + persistents + control flow; this lowering targets
// exactly that subset.
//
// State introspection from outside happens through the C++ chart
// interpreter (lib/StateChart/Interpreter.cpp), which the DAP lane
// drives. The lowered MATLAB is for native execution — drive it
// through a separate top-level driver.
//===----------------------------------------------------------------------===//

bool isIdStart(char C) {
  return std::isalpha(static_cast<unsigned char>(C)) || C == '_';
}
bool isIdCont(char C) {
  return std::isalnum(static_cast<unsigned char>(C)) || C == '_';
}

const std::set<std::string> &matlabKeywords() {
  static const std::set<std::string> K{
      "break",  "case",   "catch", "classdef", "continue", "else",
      "elseif", "end",    "for",   "function", "global",   "if",
      "otherwise", "parfor", "persistent", "return", "spmd", "switch",
      "try",    "while",  "true",  "false",    "Inf",  "NaN",
      "pi",     "eps"};
  return K;
}

std::string sanitize(const std::string &S) {
  std::string Out;
  Out.reserve(S.size());
  for (char C : S) Out += isIdCont(C) ? C : '_';
  if (Out.empty() || std::isdigit(static_cast<unsigned char>(Out.front())))
    Out = "x_" + Out;
  return Out;
}

//===----------------------------------------------------------------------===//
// Identifier maps used by the action rewriter + emit pass.
//===----------------------------------------------------------------------===//

struct ChartLayout {
  // State id → integer code (1-based). Code 0 is reserved for
  // "no substate active" / "uninitialised region".
  std::unordered_map<std::string, int> StateCode;
  // Region id → its variable name (`r_<region>`). For chart-root the
  // id is "chart_root".
  std::unordered_map<std::string, std::string> RegionVar;
  // For each OR parent (and chart-root): the list of substate ids in
  // declaration order. Indices match the order; we don't rely on
  // numeric ordering since `StateCode` is global.
  std::unordered_map<std::string, std::vector<std::string>> OrChildren;
  // For each AND parent: children sorted by executionOrder.
  std::unordered_map<std::string, std::vector<std::string>> AndChildren;
  // For each region (chart-root and every OR parent): initial substate.
  std::unordered_map<std::string, std::string> Initial;

  // Symbol classification — used by the action rewriter to decide
  // which prefix to give a bare identifier.
  std::unordered_set<std::string> Locals;
  std::unordered_set<std::string> Events;
  // Name of the chart-scoped `in()` helper. Emitted as a local
  // function; the rewriter calls it from action bodies so matlabc's
  // MATLAB→LLVM lane sees scalar arithmetic, not bool composition.
  std::string InHelper;
  // Owner state id active during the current action rewrite. Set by
  // an RAII scope (see Emitter). Empty when rewriting an expression
  // that doesn't belong to any state (e.g. a chart-root default
  // junction's cond action).
  std::string OwnerStateId;

  // Numeric code lookup.
  int codeOf(const std::string &Id) const {
    auto It = StateCode.find(Id);
    return It == StateCode.end() ? 0 : It->second;
  }
};

//===----------------------------------------------------------------------===//
// Action / guard source rewriter.
//
//   - bare local symbol      X    → l_X
//   - bare event symbol      X    → ev_X
//   - `in(stateId)`              → (r_<parent> == <code>)
//   - `emit('X')` / `emit(X)`    → ev_X = true
//   - `after(N, _)` etc.         → ((tick_count - t_<owner>) <cmp> N)
//
// String literals, comments, identifiers preceded by `.` (struct
// field access — shouldn't appear in lowered code but defensive),
// and MATLAB keywords are passed through unchanged.
//===----------------------------------------------------------------------===//

class ActionRewriter {
public:
  ActionRewriter(const Chart &C, const ChartLayout &L)
      : C_(C), L_(L) {}

  // When non-empty, every bare integer literal in the rewritten
  // source is wrapped with `<NumericCast>(...)`. Used for SV target
  // so the lowered MATLAB carries explicit fixed-width types.
  std::string NumericCast;

  std::string rewrite(const std::string &Src) const {
    std::string Out;
    Out.reserve(Src.size() + 16);
    size_t I = 0;
    char Quote = 0;
    bool PrevDot = false;
    while (I < Src.size()) {
      char C = Src[I];
      if (Quote) {
        Out += C;
        if (C == Quote) Quote = 0;
        ++I; continue;
      }
      if (C == '\'' || C == '"') {
        // Heuristic: a leading single quote right after an identifier /
        // closing-paren is the transpose operator, not a string. The
        // lowered MATLAB doesn't actually use string literals, but the
        // rewriter is also fed user-authored action source.
        bool IsString = true;
        if (C == '\'' && !Out.empty()) {
          char Last = Out.back();
          if (std::isalnum(static_cast<unsigned char>(Last)) || Last == '_' ||
              Last == ')' || Last == ']' || Last == '}')
            IsString = false;
        }
        if (IsString) Quote = C;
        Out += C; ++I; continue;
      }
      if (C == '%') {
        while (I < Src.size() && Src[I] != '\n') Out += Src[I++];
        PrevDot = false;
        continue;
      }
      // Numeric literal — consume `[+-]?\d+(\.\d+)?([eE][+-]?\d+)?`.
      // SV target wraps each with the configured cast (e.g.
      // `int16(...)`) so matlabc's SV pipeline sees explicit widths
      // on every literal. Software target passes them through
      // unchanged.
      if (std::isdigit(static_cast<unsigned char>(C)) && !PrevDot) {
        size_t Start = I;
        while (I < Src.size() &&
               std::isdigit(static_cast<unsigned char>(Src[I]))) ++I;
        bool IsFloat = false;
        if (I < Src.size() && Src[I] == '.' && I + 1 < Src.size() &&
            std::isdigit(static_cast<unsigned char>(Src[I + 1]))) {
          IsFloat = true;
          ++I;
          while (I < Src.size() &&
                 std::isdigit(static_cast<unsigned char>(Src[I]))) ++I;
        }
        if (I < Src.size() && (Src[I] == 'e' || Src[I] == 'E')) {
          IsFloat = true;
          ++I;
          if (I < Src.size() && (Src[I] == '+' || Src[I] == '-')) ++I;
          while (I < Src.size() &&
                 std::isdigit(static_cast<unsigned char>(Src[I]))) ++I;
        }
        std::string Lit = Src.substr(Start, I - Start);
        if (!NumericCast.empty() && !IsFloat) {
          Out += NumericCast;
          Out += "(";
          Out += Lit;
          Out += ")";
        } else {
          Out += Lit;
        }
        continue;
      }
      if (C == '.' && I + 1 < Src.size() && Src[I + 1] != '.') {
        Out += C; ++I; PrevDot = true; continue;
      }
      if (isIdStart(C)) {
        size_t Start = I;
        while (I < Src.size() && isIdCont(Src[I])) ++I;
        std::string Id = Src.substr(Start, I - Start);

        // `in(stateId)` — rewrite inline. Strict form expected:
        // `in(X)` or `in('X')` or `in("X")`. Unknown forms fall
        // through to the generic identifier branch.
        if (!PrevDot && Id == "in") {
          if (auto Body = parseIdArgCall(Src, I)) {
            auto Code = codeOfState(Body->Name);
            if (Code) {
              std::string ParentVar = parentRegionVar(Body->Name);
              if (!ParentVar.empty()) {
                if (NumericCast.empty()) {
                  // Software target: route through the chart-scoped
                  // helper so the call site sees a scalar return.
                  // matlabc's MATLAB→LLVM lane refuses `add` of two
                  // logicals; helper-returned scalars compose.
                  Out += L_.InHelper;
                  Out += "(";
                  Out += std::to_string(*Code);
                  Out += ", ";
                  Out += ParentVar;
                  Out += ")";
                } else {
                  // SV target: emit a typed comparison inline + wrap
                  // the bool result in `intW(...)` so it composes
                  // with arithmetic (`a + b` between bools yields
                  // f64 which the SV pipeline rejects). matlabc's
                  // synth lane handles `intW(bool)` as a 0/1 widen.
                  Out += NumericCast;
                  Out += "(";
                  Out += ParentVar;
                  Out += " == ";
                  Out += NumericCast;
                  Out += "(";
                  Out += std::to_string(*Code);
                  Out += "))";
                }
                I = Body->Pos;
                PrevDot = false;
                continue;
              }
            }
          }
        }
        // `emit('X')` — rewrite to local event-flag write. The
        // chart_tick body holds `ev_X` in scope (as both the input
        // arg and the in-action mutation slot).
        if (!PrevDot && Id == "emit") {
          if (auto Body = parseIdArgCall(Src, I)) {
            Out += "ev_";
            Out += Body->Name;
            Out += " = true";
            I = Body->Pos;
            PrevDot = false;
            continue;
          }
        }
        // Temporal operators: `after(N, unit)` etc.
        if (!PrevDot && !L_.OwnerStateId.empty() &&
            (Id == "after" || Id == "before" ||
             Id == "at" || Id == "every")) {
          if (auto N = parseTemporalCall(Src, I)) {
            std::string Delta = "(tick_count - t_" +
                                sanitize(L_.OwnerStateId) + ")";
            std::string Expr;
            if      (Id == "after")  Expr = "(" + Delta + " >= " + *N + ")";
            else if (Id == "before") Expr = "(" + Delta + " <  " + *N + ")";
            else if (Id == "at")     Expr = "(" + Delta + " == " + *N + ")";
            else                     Expr =
                "((" + Delta + " > 0) && (mod(" + Delta + ", " + *N +
                ") == 0))";
            Out += Expr;
            I = parseTemporalEnd(Src, I);
            PrevDot = false;
            continue;
          }
        }

        if (PrevDot || matlabKeywords().count(Id)) {
          Out += Id;
        } else if (L_.Locals.count(Id)) {
          Out += "l_";
          Out += Id;
        } else if (L_.Events.count(Id)) {
          Out += "ev_";
          Out += Id;
        } else {
          Out += Id;
        }
        PrevDot = false;
        continue;
      }
      Out += C; ++I;
      if (!std::isspace(static_cast<unsigned char>(C))) PrevDot = false;
    }
    return Out;
  }

private:
  const Chart &C_;
  const ChartLayout &L_;

  struct ParsedArg {
    std::string Name;
    size_t Pos;
  };

  // Parse `(<ident-or-string>)` starting at `Pos` (which sits just
  // past the call identifier). Advances `Pos` to one past the `)`
  // on success. Returns nullopt on shape mismatch.
  std::optional<ParsedArg> parseIdArgCall(const std::string &Src,
                                          size_t Pos) const {
    size_t P = Pos;
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    if (P >= Src.size() || Src[P] != '(') return std::nullopt;
    ++P;
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    std::string Name;
    if (P < Src.size() && (Src[P] == '\'' || Src[P] == '"')) {
      char Q = Src[P++];
      size_t S = P;
      while (P < Src.size() && Src[P] != Q) ++P;
      if (P >= Src.size()) return std::nullopt;
      Name = Src.substr(S, P - S);
      ++P;
    } else if (P < Src.size() && isIdStart(Src[P])) {
      size_t S = P;
      while (P < Src.size() && isIdCont(Src[P])) ++P;
      Name = Src.substr(S, P - S);
    } else {
      return std::nullopt;
    }
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    if (P >= Src.size() || Src[P] != ')') return std::nullopt;
    ++P;
    return ParsedArg{std::move(Name), P};
  }

  // Parse `(N[, unit])` — returns the literal `N` text on success.
  std::optional<std::string> parseTemporalCall(const std::string &Src,
                                               size_t Pos) const {
    size_t P = Pos;
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    if (P >= Src.size() || Src[P] != '(') return std::nullopt;
    ++P;
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    size_t S = P;
    if (P < Src.size() && (Src[P] == '+' || Src[P] == '-')) ++P;
    while (P < Src.size() && std::isdigit((unsigned char)Src[P])) ++P;
    if (P == S) return std::nullopt;
    std::string N = Src.substr(S, P - S);
    // Consume optional `, unit` and trailing `)`.
    while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    if (P < Src.size() && Src[P] == ',') {
      ++P;
      while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
      while (P < Src.size() && isIdCont(Src[P])) ++P;
      while (P < Src.size() && std::isspace((unsigned char)Src[P])) ++P;
    }
    if (P >= Src.size() || Src[P] != ')') return std::nullopt;
    return N;
  }

  size_t parseTemporalEnd(const std::string &Src, size_t Pos) const {
    size_t P = Pos;
    while (P < Src.size() && Src[P] != ')') ++P;
    return P + 1;
  }

  std::optional<int> codeOfState(const std::string &Id) const {
    auto It = L_.StateCode.find(Id);
    if (It == L_.StateCode.end()) return std::nullopt;
    return It->second;
  }

  std::string parentRegionVar(const std::string &StateId) const {
    const ChartState *S = C_.findState(StateId);
    if (!S) return {};
    std::string Parent = S->ParentId;
    while (true) {
      const ChartState *P = Parent.empty() ? nullptr : C_.findState(Parent);
      bool ParentIsAnd =
          P && P->Decomp == Decomposition::And;
      if (!ParentIsAnd) {
        auto It = L_.RegionVar.find(Parent.empty() ? "chart_root" : Parent);
        return It == L_.RegionVar.end() ? std::string() : It->second;
      }
      Parent = P->ParentId;
    }
  }
};

//===----------------------------------------------------------------------===//
// Lowering driver.
//===----------------------------------------------------------------------===//

class Emitter {
public:
  Emitter(const Chart &C, DiagnosticEngine &Diag,
          const LoweringOptions &Opts)
      : C_(C), Diag_(Diag), Opts_(Opts), Rewriter_(C, L_) {}

  std::optional<LoweringResult> emit() {
    buildLayout();
    LoweringResult R;
    R.InitFunction = "";                                   // no longer used
    R.TickFunction = sanitize(C_.Name) + "_tick";

    bool UsesIn = chartUsesInPredicate();
    std::ostringstream OS;
    emitHeader(OS);
    if (Opts_.Target == LoweringTarget::Software) {
      emitDemoDriver(OS);
    } else {
      // SV target: emit a small calls-once harness so matlabc still
      // sees a script-with-function (its SV pipeline ignores the
      // top-level for module emission).
      emitSvHarness(OS);
    }
    emitTickFunction(OS);
    // The `in()` helper is only needed for software target (its
    // function-returned scalar dodges matlabc's MATLAB→LLVM
    // bool-composition gap). For SV, the rewriter emits inline
    // typed comparisons that the SV pipeline lowers directly, so
    // the helper is dead code and must not be emitted (matlabc
    // would still try to type its `any` params and fail).
    if (UsesIn && Opts_.Target == LoweringTarget::Software)
      emitInHelper(OS);
    // Note: state-name lookup helper omitted on purpose — it would
    // emit `name = '...'` string literals which matlabc's MATLAB→
    // LLVM lane doesn't lower. Use the header's state-code legend
    // for human-readable mapping, or call into the chart-DAP
    // listStates request.
    R.MatlabSource = OS.str();
    return R;
  }

  // Stateless `in(code, region_val)` helper — returns 1 when the
  // region holds the requested code, else 0. Routed through a
  // function so the call site sees a scalar return (not a bool
  // expression), which matlabc's emit-llvm composes via arithmetic.
  // For SV target, the result + literals are wrapped in `intW(...)`
  // so the synthesizable pipeline can pick a width.
  void emitInHelper(std::ostream &OS) {
    OS << "function out = " << L_.InHelper << "(code, region_val)\n";
    OS << "  out = " << codeText(0) << ";\n";
    OS << "  if region_val == code\n";
    OS << "    out = " << codeText(1) << ";\n";
    OS << "  end\n";
    OS << "end\n";
  }

private:
  const Chart &C_;
  [[maybe_unused]] DiagnosticEngine &Diag_;
  const LoweringOptions &Opts_;
  ChartLayout L_;
  ActionRewriter Rewriter_;

  // Owner-scope RAII for action rewriting context.
  struct OwnerScope {
    ChartLayout &L;
    std::string Prev;
    OwnerScope(ChartLayout &Layout, std::string New)
        : L(Layout), Prev(Layout.OwnerStateId) {
      L.OwnerStateId = std::move(New);
    }
    ~OwnerScope() { L.OwnerStateId = std::move(Prev); }
  };

  std::string rewrite(const std::string &Src) const {
    return Rewriter_.rewrite(Src);
  }

  // Textual form of an integer constant. Wraps with `int<W>(N)` for
  // SV target so matlabc's type inference picks the right width.
  std::string codeText(int N) const {
    std::string S = std::to_string(N);
    if (Opts_.Target == LoweringTarget::SystemVerilog)
      return "int" + std::to_string(Opts_.IntegerWidth) + "(" + S + ")";
    return S;
  }

  void buildLayout() {
    L_.InHelper = sanitize(C_.Name) + "_in_";
    int Code = 1;
    for (auto &P : C_.States) L_.StateCode[P.first] = Code++;

    for (auto &N : C_.Sig.Inputs)  L_.Locals.insert(N);
    for (auto &N : C_.Sig.Outputs) L_.Locals.insert(N);
    for (auto &S : C_.Symbols.Data) L_.Locals.insert(S.Name);
    for (auto &S : C_.Symbols.Events) L_.Events.insert(S.Name);

    L_.RegionVar["chart_root"] = "r_chart_root";
    for (auto &P : C_.States) {
      const ChartState &S = P.second;
      if (S.Decomp == Decomposition::Or && !S.ChildStateIds.empty())
        L_.RegionVar[S.Id] = "r_" + sanitize(S.Id);
      if (S.Decomp == Decomposition::And) {
        std::vector<std::string> Kids = S.ChildStateIds;
        std::stable_sort(Kids.begin(), Kids.end(),
            [&](const std::string &A, const std::string &B) {
              int Ai = 0, Bi = 0;
              if (auto *Sa = C_.findState(A))
                Ai = Sa->ExecutionOrder.value_or(0);
              if (auto *Sb = C_.findState(B))
                Bi = Sb->ExecutionOrder.value_or(0);
              return Ai < Bi;
            });
        L_.AndChildren[S.Id] = std::move(Kids);
      }
    }

    // OR-children buckets + initial substates.
    L_.OrChildren["chart_root"] = C_.RootStateIds;
    L_.Initial["chart_root"] = initialSubstate("chart_root", C_.RootStateIds);
    for (auto &P : C_.States) {
      const ChartState &S = P.second;
      if (S.Decomp != Decomposition::Or || S.ChildStateIds.empty()) continue;
      L_.OrChildren[S.Id] = S.ChildStateIds;
      L_.Initial[S.Id] = initialSubstate(S.Id, S.ChildStateIds);
    }
  }

  std::string initialSubstate(const std::string &ParentId,
                              const std::vector<std::string> &Kids) const {
    for (auto &K : Kids) {
      const auto *S = C_.findState(K);
      if (S && S->IsInitial) return K;
    }
    // Default-junction fallback.
    std::string Look = (ParentId == "chart_root") ? std::string() : ParentId;
    for (auto &P : C_.Junctions) {
      const ChartJunction &J = P.second;
      if (J.Kind != JunctionKind::Default) continue;
      if (J.ParentId != Look) continue;
      for (auto &T : C_.Transitions)
        if (T.SourceId == J.Id) return T.DestId;
    }
    return Kids.empty() ? std::string() : Kids.front();
  }

  std::vector<const Transition *> outgoingFrom(const std::string &Id) const {
    std::vector<const Transition *> Out;
    for (auto &T : C_.Transitions)
      if (T.SourceId == Id) Out.push_back(&T);
    std::stable_sort(Out.begin(), Out.end(),
        [](const Transition *A, const Transition *B) {
          return A->Priority < B->Priority;
        });
    return Out;
  }

  bool isOrParent(const std::string &Id) const {
    const auto *S = C_.findState(Id);
    return S && S->Decomp == Decomposition::Or && !S->ChildStateIds.empty();
  }

  std::string regionOwner(const std::string &StateId) const {
    const ChartState *S = C_.findState(StateId);
    if (!S) return "chart_root";
    std::string Parent = S->ParentId;
    while (true) {
      if (Parent.empty()) return "chart_root";
      const ChartState *P = C_.findState(Parent);
      if (!P) return "chart_root";
      if (P->Decomp == Decomposition::And) {
        Parent = P->ParentId;
        continue;
      }
      return Parent;
    }
  }

  std::string lcaOf(const std::string &A, const std::string &B) const {
    auto chain = [&](const std::string &Id) {
      std::vector<std::string> Out;
      std::string Cur = Id;
      while (!Cur.empty()) {
        Out.push_back(Cur);
        const auto *S = C_.findState(Cur);
        const auto *J = C_.findJunction(Cur);
        Cur = S ? S->ParentId : (J ? J->ParentId : std::string());
      }
      return Out;
    };
    std::unordered_set<std::string> AS;
    for (auto &S : chain(A)) AS.insert(S);
    for (auto &S : chain(B)) if (AS.count(S)) return S;
    return {};
  }

  //===----- emission helpers ------------------------------------------------

  void emitHeader(std::ostream &OS) {
    OS << "% Auto-generated by matlabc — DO NOT EDIT.\n";
    OS << "% Chart: " << C_.Name << "\n";
    OS << "%\n% State codes (1-based; 0 = uninitialised):\n";
    std::vector<std::pair<std::string, int>> Order;
    for (auto &P : L_.StateCode) Order.emplace_back(P.first, P.second);
    std::sort(Order.begin(), Order.end(),
              [](const auto &A, const auto &B) { return A.second < B.second; });
    for (auto &P : Order)
      OS << "%   " << P.second << " = " << P.first << "\n";
    OS << "\n";
  }

  // Demo driver: a few super-step ticks with deterministic input,
  // printing each output. matlabc -emit-llvm / -emit-c need at least
  // one top-level statement to treat the file as a script-with-
  // functions. Users replace the loop body for their own exercise.
  void emitDemoDriver(std::ostream &OS) {
    OS << "% --- Demo driver: 5 ticks. Replace with your own driver. ---\n";
    std::ostringstream Call;
    Call << sanitize(C_.Name) << "_tick(";
    bool First = true;
    for (auto &N : C_.Sig.Inputs) {
      (void)N;
      if (!First) Call << ", "; First = false;
      Call << "0";
    }
    for (auto &E : C_.Symbols.Events) {
      (void)E;
      if (!First) Call << ", "; First = false;
      Call << "false";
    }
    Call << ")";
    if (C_.Sig.Outputs.empty()) {
      OS << "for k = 1:5\n  " << Call.str() << ";\nend\n";
    } else if (C_.Sig.Outputs.size() == 1) {
      OS << "for k = 1:5\n  disp(" << Call.str() << ");\nend\n";
    } else {
      OS << "for k = 1:5\n  [";
      for (size_t I = 0; I < C_.Sig.Outputs.size(); ++I) {
        if (I) OS << ", ";
        OS << "out" << (I + 1);
      }
      OS << "] = " << Call.str() << ";\n";
      for (size_t I = 0; I < C_.Sig.Outputs.size(); ++I)
        OS << "  disp(out" << (I + 1) << ");\n";
      OS << "end\n";
    }
    OS << "\n";
  }

  // SV-target harness — minimum top-level statement so matlabc's
  // SV pipeline treats the file as script-with-function. The call
  // also binds parameter types: int32 for the inputs, logical for
  // the event flags. SV emission ignores the top-level statements
  // and emits a module from the function body only.
  void emitSvHarness(std::ostream &OS) {
    OS << "% --- SV harness: drives one tick with typed args for type inference. ---\n";
    std::string IntT = "int" + std::to_string(Opts_.IntegerWidth);
    std::ostringstream Call;
    Call << sanitize(C_.Name) << "_tick(";
    bool First = true;
    for (auto &N : C_.Sig.Inputs) {
      (void)N;
      if (!First) Call << ", "; First = false;
      Call << IntT << "(0)";
    }
    for (auto &E : C_.Symbols.Events) {
      (void)E;
      if (!First) Call << ", "; First = false;
      Call << "false";
    }
    Call << ");";
    OS << Call.str() << "\n\n";
  }

  // The chart's tick function. One large function with everything
  // inlined: init block, first-call entry chain, super-step loop,
  // output assignment.
  void emitTickFunction(std::ostream &OS) {
    if (Opts_.Target == LoweringTarget::SystemVerilog) {
      emitTickFunctionSv(OS);
      return;
    }
    // Signature.
    OS << "function ";
    if (!C_.Sig.Outputs.empty()) {
      if (C_.Sig.Outputs.size() == 1) {
        OS << "out_" << C_.Sig.Outputs.front();
      } else {
        OS << "[";
        for (size_t I = 0; I < C_.Sig.Outputs.size(); ++I) {
          if (I) OS << ", ";
          OS << "out_" << C_.Sig.Outputs[I];
        }
        OS << "]";
      }
      OS << " = ";
    }
    OS << sanitize(C_.Name) << "_tick(";
    bool First = true;
    for (auto &N : C_.Sig.Inputs) {
      if (!First) OS << ", "; First = false;
      OS << "in_" << N;
    }
    for (auto &E : C_.Symbols.Events) {
      if (!First) OS << ", "; First = false;
      OS << "ev_" << E.Name;
    }
    OS << ")\n";

    // Persistent declarations — one variable per line. matlabc's
    // emit-c lane has a bug where a multi-variable `persistent a b c;`
    // line emits a display of the last variable as an unsuppressed
    // expression statement. One-var-per-line dodges it.
    OS << "  persistent init_done;\n";
    OS << "  persistent tick_count;\n";
    for (auto &P : L_.RegionVar)
      OS << "  persistent " << P.second << ";\n";
    for (auto &P : C_.States)
      if (P.second.HasHistory)
        OS << "  persistent h_" << sanitize(P.first) << ";\n";
    for (auto &P : C_.States)
      OS << "  persistent t_" << sanitize(P.first) << ";\n";
    for (auto &N : L_.Locals)
      OS << "  persistent l_" << N << ";\n";

    // isempty-init block: zero everything, seed locals from the
    // symbol table's `initial`, descend the initial entry chain.
    OS << "  if isempty(init_done)\n";
    OS << "    init_done = 0;\n";
    OS << "    tick_count = 0;\n";
    for (auto &P : L_.RegionVar) OS << "    " << P.second << " = 0;\n";
    for (auto &P : C_.States)
      if (P.second.HasHistory)
        OS << "    h_" << sanitize(P.first) << " = 0;\n";
    for (auto &P : C_.States)
      OS << "    t_" << sanitize(P.first) << " = 0;\n";
    for (auto &S : C_.Symbols.Data) {
      OS << "    l_" << S.Name << " = ";
      OS << (S.Initial.empty() ? "0" : Rewriter_.rewrite(S.Initial)) << ";\n";
    }
    for (auto &N : C_.Sig.Inputs)
      if (!hasDataSymbol(N)) OS << "    l_" << N << " = 0;\n";
    for (auto &N : C_.Sig.Outputs)
      if (!hasDataSymbol(N)) OS << "    l_" << N << " = 0;\n";
    OS << "  end\n";

    // Bind inputs into the chart's local scope so user actions can
    // read them as bare names (already rewritten to `l_*`).
    for (auto &N : C_.Sig.Inputs)
      OS << "  l_" << N << " = in_" << N << ";\n";

    // First-call entry chain.
    OS << "  if init_done == 0\n";
    emitEnterChain(OS, L_.Initial["chart_root"], "    ");
    OS << "    init_done = 1;\n";
    OS << "  end\n";

    // Advance tick counter for temporal operators.
    OS << "  tick_count = tick_count + 1;\n";

    // Super-step loop. Using `while` over `for + break` because
    // matlabc's `-emit-c` lane has a bug lowering single-stmt
    // `if cond, break; end` (references an undeclared
    // `__did_break`). Saturation surfaces as `iter == MaxIterations`
    // — the caller can inspect by adding an extra return value.
    OS << "  fired = true;\n";
    OS << "  iter = 0;\n";
    OS << "  while fired\n";
    OS << "    if iter >= " << C_.MaxIterations << "\n";
    OS << "      fired = false;\n";
    OS << "    else\n";
    OS << "      iter = iter + 1;\n";
    OS << "      fired = false;\n";
    emitStepFromRegion(OS, "chart_root", "      ");
    OS << "    end\n";
    OS << "  end\n";

    // Output assignment.
    for (auto &N : C_.Sig.Outputs)
      OS << "  out_" << N << " = l_" << N << ";\n";
    OS << "end\n\n";
  }

  bool hasDataSymbol(const std::string &Name) const {
    for (auto &S : C_.Symbols.Data) if (S.Name == Name) return true;
    return false;
  }

  //===-- SV target -----------------------------------------------------===
  // matlabc's SV pipeline needs each persistent to carry its own
  // canonical `if isempty(X), X = const; end` initializer (it maps
  // them to power-on reset values), it rejects data-dependent while-
  // loops, and it rejects floating-point types on function params.
  // Emit one-pass tick form: a single transition-check pass per
  // invocation, no inner super-step loop. The chart's user-visible
  // behaviour is: each call advances the FSM by at most one
  // transition per region. Typical Moore/Mealy synthesizable charts.
  void emitTickFunctionSv(std::ostream &OS) {
    std::string IntT = "int" + std::to_string(Opts_.IntegerWidth);
    Rewriter_.NumericCast = IntT;
    // Signature.
    OS << "function ";
    if (!C_.Sig.Outputs.empty()) {
      if (C_.Sig.Outputs.size() == 1) {
        OS << "out_" << C_.Sig.Outputs.front();
      } else {
        OS << "[";
        for (size_t I = 0; I < C_.Sig.Outputs.size(); ++I) {
          if (I) OS << ", ";
          OS << "out_" << C_.Sig.Outputs[I];
        }
        OS << "]";
      }
      OS << " = ";
    }
    OS << sanitize(C_.Name) << "_tick(";
    bool First = true;
    for (auto &N : C_.Sig.Inputs) {
      if (!First) OS << ", "; First = false;
      OS << "in_" << N;
    }
    for (auto &E : C_.Symbols.Events) {
      if (!First) OS << ", "; First = false;
      OS << "ev_" << E.Name;
    }
    OS << ")\n";

    auto persIsempty = [&](const std::string &Name, const std::string &Init) {
      OS << "  persistent " << Name << ";\n";
      OS << "  if isempty(" << Name << "), " << Name << " = " << Init
         << "; end\n";
    };

    persIsempty("init_done", IntT + "(0)");
    persIsempty("tick_count", IntT + "(0)");
    for (auto &P : L_.RegionVar) persIsempty(P.second, IntT + "(0)");
    for (auto &P : C_.States)
      if (P.second.HasHistory)
        persIsempty("h_" + sanitize(P.first), IntT + "(0)");
    for (auto &P : C_.States)
      persIsempty("t_" + sanitize(P.first), IntT + "(0)");
    for (auto &S : C_.Symbols.Data)
      persIsempty("l_" + S.Name,
                  IntT + "(" + (S.Initial.empty() ? "0" : S.Initial) + ")");
    for (auto &N : C_.Sig.Inputs)
      if (!hasDataSymbol(N)) persIsempty("l_" + N, IntT + "(0)");
    for (auto &N : C_.Sig.Outputs)
      if (!hasDataSymbol(N)) persIsempty("l_" + N, IntT + "(0)");

    // Bind inputs (already typed by caller).
    for (auto &N : C_.Sig.Inputs)
      OS << "  l_" << N << " = in_" << N << ";\n";

    // First-call entry chain: gated by init_done; identical chain to
    // software target. The chain writes the initial-substate code
    // into the region slot + runs entry actions.
    OS << "  if init_done == " << IntT << "(0)\n";
    emitEnterChain(OS, L_.Initial["chart_root"], "    ");
    OS << "    init_done = " << IntT << "(1);\n";
    OS << "  end\n";

    OS << "  tick_count = tick_count + " << IntT << "(1);\n";

    // Single-pass transition evaluation — one shot, no while loop.
    // SV pipelines model this as one clock edge per chart_tick call.
    OS << "  fired = false;\n";
    emitStepFromRegion(OS, "chart_root", "  ");

    for (auto &N : C_.Sig.Outputs)
      OS << "  out_" << N << " = l_" << N << ";\n";
    OS << "end\n\n";
  }

  // Scan every action / guard body for an `in(` call so we only emit
  // the in-helper when something actually calls it.
  bool chartUsesInPredicate() const {
    auto scan = [](const std::string &S) {
      size_t I = 0;
      while ((I = S.find("in", I)) != std::string::npos) {
        bool LeftOk = (I == 0) ||
                      !isIdCont(S[I - 1]);
        size_t After = I + 2;
        while (After < S.size() &&
               std::isspace(static_cast<unsigned char>(S[After]))) ++After;
        if (LeftOk && After < S.size() && S[After] == '(') return true;
        I += 2;
      }
      return false;
    };
    for (auto &P : C_.States) {
      const ChartState &S = P.second;
      if (scan(S.Entry.Source) || scan(S.During.Source) ||
          scan(S.Exit.Source))
        return true;
      for (auto &OE : S.OnEvent)
        if (scan(OE.second.Source)) return true;
    }
    for (auto &T : C_.Transitions) {
      if (scan(T.Label.Guard) || scan(T.Label.CondAction) ||
          scan(T.Label.TransAction))
        return true;
    }
    return false;
  }

  // Emit the entry chain for `Sid`: set its parent region slot (if
  // any), stamp entry-time, run entry action, then descend into the
  // initial substate for OR parents / every child for AND parents.
  // History-aware: when entering an OR-parent flagged with hasHistory
  // and h_<id> != 0, dispatch into the previously-active substate.
  void emitEnterChain(std::ostream &OS, const std::string &Sid,
                      const std::string &Pad) {
    if (Sid.empty()) return;
    const ChartState *S = C_.findState(Sid);
    if (!S) return;
    OwnerScope Owner(L_, Sid);
    // Mark parent region's active slot (OR-parent or chart-root only).
    std::string ParentRegion = regionOwner(Sid);
    OS << Pad << L_.RegionVar.at(ParentRegion) << " = "
       << codeText(L_.codeOf(Sid)) << ";\n";
    OS << Pad << "t_" << sanitize(Sid) << " = tick_count;\n";
    if (!S->Entry.empty())
      OS << Pad << rewrite(S->Entry.Source) << ";\n";
    if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
      if (S->HasHistory) {
        OS << Pad << "if h_" << sanitize(Sid) << " ~= 0\n";
        for (auto &Cid : S->ChildStateIds) {
          OS << Pad << "  if h_" << sanitize(Sid) << " == "
             << codeText(L_.codeOf(Cid)) << "\n";
          emitEnterChain(OS, Cid, Pad + "    ");
          OS << Pad << "  end\n";
        }
        OS << Pad << "else\n";
        emitEnterChain(OS, L_.Initial.at(Sid), Pad + "  ");
        OS << Pad << "end\n";
      } else {
        emitEnterChain(OS, L_.Initial.at(Sid), Pad);
      }
    } else if (S->Decomp == Decomposition::And) {
      for (auto &K : L_.AndChildren.at(Sid)) emitEnterChain(OS, K, Pad);
    }
  }

  // Emit the exit chain for `Sid`: recursively exit active substates
  // first, save history if the parent has it, then run own exit
  // action. The parent region's slot is NOT cleared here — the
  // caller resets it (typically by writing the destination's code).
  void emitExitChain(std::ostream &OS, const std::string &Sid,
                     const std::string &Pad) {
    const ChartState *S = C_.findState(Sid);
    if (!S) return;
    OwnerScope Owner(L_, Sid);
    if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
      // Walk via the currently-active code in this region.
      std::string Var = L_.RegionVar.at(Sid);
      for (auto &Cid : S->ChildStateIds) {
        OS << Pad << "if " << Var << " == " << codeText(L_.codeOf(Cid)) << "\n";
        emitExitChain(OS, Cid, Pad + "  ");
        OS << Pad << "end\n";
      }
      if (S->HasHistory)
        OS << Pad << "h_" << sanitize(Sid) << " = " << Var << ";\n";
      OS << Pad << Var << " = 0;\n";
    } else if (S->Decomp == Decomposition::And) {
      auto Kids = L_.AndChildren.at(Sid);
      // Exit in reverse exec order.
      std::reverse(Kids.begin(), Kids.end());
      for (auto &K : Kids) emitExitChain(OS, K, Pad);
    }
    if (!S->Exit.empty())
      OS << Pad << rewrite(S->Exit.Source) << ";\n";
  }

  // Step a region: select the active substate via the region's
  // variable, dispatch into its transitions / on-event / during /
  // recursive AND descent.
  void emitStepFromRegion(std::ostream &OS, const std::string &RegionId,
                          const std::string &Pad) {
    auto It = L_.OrChildren.find(RegionId);
    if (It == L_.OrChildren.end() || It->second.empty()) return;
    std::string Var = L_.RegionVar.at(RegionId);
    for (auto &Cid : It->second) {
      OS << Pad << "if " << Var << " == " << codeText(L_.codeOf(Cid)) << "\n";
      emitActiveSubstateBody(OS, Cid, Pad + "  ");
      OS << Pad << "end\n";
    }
  }

  void emitActiveSubstateBody(std::ostream &OS, const std::string &Sid,
                              const std::string &Pad) {
    const ChartState *S = C_.findState(Sid);
    if (!S) return;
    OwnerScope Owner(L_, Sid);
    // Outgoing transitions in priority order.
    for (auto *T : outgoingFrom(Sid)) emitTransition(OS, *T, Pad);
    // on-event handlers (only for known events).
    for (auto &OE : S->OnEvent) {
      if (!L_.Events.count(OE.first)) continue;
      OS << Pad << "if ~fired && ev_" << OE.first << "\n";
      OS << Pad << "  " << rewrite(OE.second.Source) << ";\n";
      OS << Pad << "end\n";
    }
    // During action.
    if (!S->During.empty()) {
      OS << Pad << "if ~fired\n";
      OS << Pad << "  " << rewrite(S->During.Source) << ";\n";
      OS << Pad << "end\n";
    }
    // Recurse into substates.
    if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
      emitStepFromRegion(OS, Sid, Pad);
    } else if (S->Decomp == Decomposition::And) {
      for (auto &K : L_.AndChildren.at(Sid))
        emitActiveSubstateBody(OS, K, Pad);
    }
  }

  void emitTransition(std::ostream &OS, const Transition &T,
                      const std::string &Pad) {
    // Cond expression. Event-gated transitions read the `ev_<E>`
    // bool from the chart_tick scope.
    std::string Cond;
    bool HasEvent = !T.Label.Event.empty() && L_.Events.count(T.Label.Event);
    bool HasGuard = !T.Label.Guard.empty();
    if (HasEvent && HasGuard)
      Cond = "ev_" + T.Label.Event + " && (" +
             rewrite(T.Label.Guard) + ")";
    else if (HasEvent)
      Cond = "ev_" + T.Label.Event;
    else if (HasGuard)
      Cond = "(" + rewrite(T.Label.Guard) + ")";
    else
      Cond = "true";

    OS << Pad << "if ~fired && " << Cond << "\n";
    if (!T.Label.CondAction.empty())
      OS << Pad << "  " << rewrite(T.Label.CondAction) << ";\n";

    bool IsInner = T.Kind == TransitionKind::Inner;
    if (!IsInner) {
      const ChartState *Src = C_.findState(T.SourceId);
      const ChartState *Dst = C_.findState(T.DestId);
      if (Src && Dst) {
        std::string LCA = lcaOf(T.SourceId, T.DestId);
        // Exit chain: walk src up to (but not including) LCA. Each
        // step runs its exit action and clears its region slot if
        // we're crossing an OR parent boundary.
        std::string Cur = T.SourceId;
        while (!Cur.empty() && Cur != LCA) {
          emitExitChain(OS, Cur, Pad + "  ");
          const ChartState *S = C_.findState(Cur);
          Cur = S ? S->ParentId : std::string();
        }
        // Trans action.
        if (!T.Label.TransAction.empty())
          OS << Pad << "  " << rewrite(T.Label.TransAction) << ";\n";
        // Enter chain: build path LCA→dst and emit shallow entry on
        // each level above dst, then full enter at dst (which
        // recurses into substates).
        std::vector<std::string> EnterChain;
        Cur = T.DestId;
        while (!Cur.empty() && Cur != LCA) {
          EnterChain.push_back(Cur);
          const ChartState *S = C_.findState(Cur);
          Cur = S ? S->ParentId : std::string();
        }
        std::reverse(EnterChain.begin(), EnterChain.end());
        for (size_t I = 0; I + 1 < EnterChain.size(); ++I) {
          const ChartState *S = C_.findState(EnterChain[I]);
          if (!S) continue;
          std::string Owner = regionOwner(EnterChain[I]);
          OS << Pad << "  " << L_.RegionVar.at(Owner) << " = "
             << codeText(L_.codeOf(EnterChain[I])) << ";\n";
          OS << Pad << "  t_" << sanitize(EnterChain[I])
             << " = tick_count;\n";
          if (!S->Entry.empty()) {
            OwnerScope OS_(L_, EnterChain[I]);
            OS << Pad << "  " << rewrite(S->Entry.Source) << ";\n";
          }
        }
        // Deepest level (dst) — full enter, recurses into substates.
        emitEnterChain(OS, T.DestId, Pad + "  ");
      } else if (Dst) {
        // Source was a junction — just trans + enter.
        if (!T.Label.TransAction.empty())
          OS << Pad << "  " << rewrite(T.Label.TransAction) << ";\n";
        emitEnterChain(OS, T.DestId, Pad + "  ");
      } else {
        // Destination is itself a junction — emit chain.
        if (!T.Label.TransAction.empty())
          OS << Pad << "  " << rewrite(T.Label.TransAction) << ";\n";
        emitJunctionChain(OS, T.DestId, Pad + "  ", 0);
      }
    } else {
      // Inner — only the trans action runs.
      if (!T.Label.TransAction.empty())
        OS << Pad << "  " << rewrite(T.Label.TransAction) << ";\n";
    }
    OS << Pad << "  fired = true;\n";
    OS << Pad << "end\n";
  }

  void emitJunctionChain(std::ostream &OS, const std::string &JctId,
                         const std::string &Pad, int Depth) {
    if (Depth > 16) {
      OS << Pad << "% junction chain depth exceeded\n";
      return;
    }
    const ChartJunction *J = C_.findJunction(JctId);
    if (!J) return;
    if (J->Kind == JunctionKind::History) {
      const ChartState *Parent = C_.findState(J->ParentId);
      if (!Parent) return;
      OS << Pad << "if h_" << sanitize(J->ParentId) << " ~= 0\n";
      for (auto &Cid : Parent->ChildStateIds) {
        OS << Pad << "  if h_" << sanitize(J->ParentId) << " == "
           << codeText(L_.codeOf(Cid)) << "\n";
        emitEnterChain(OS, Cid, Pad + "    ");
        OS << Pad << "  end\n";
      }
      OS << Pad << "else\n";
      emitEnterChain(OS, L_.Initial.at(J->ParentId), Pad + "  ");
      OS << Pad << "end\n";
      return;
    }
    std::vector<const Transition *> Outs;
    for (auto &T : C_.Transitions)
      if (T.SourceId == JctId) Outs.push_back(&T);
    std::stable_sort(Outs.begin(), Outs.end(),
        [](const Transition *A, const Transition *B) {
          return A->Priority < B->Priority;
        });
    bool First = true;
    for (auto *T : Outs) {
      std::string Cond = T->Label.Guard.empty()
                             ? std::string("true")
                             : "(" + rewrite(T->Label.Guard) + ")";
      OS << Pad << (First ? "if " : "elseif ") << Cond << "\n";
      First = false;
      if (!T->Label.CondAction.empty())
        OS << Pad << "  " << rewrite(T->Label.CondAction) << ";\n";
      if (!T->Label.TransAction.empty())
        OS << Pad << "  " << rewrite(T->Label.TransAction) << ";\n";
      if (C_.findState(T->DestId))
        emitEnterChain(OS, T->DestId, Pad + "  ");
      else
        emitJunctionChain(OS, T->DestId, Pad + "  ", Depth + 1);
    }
    if (!First) OS << Pad << "end\n";
  }

  // Stateless human-readable name-from-code helper. Useful from a
  // driver script for log lines. Emitted as a local function.
  void emitNameLookup(std::ostream &OS) {
    OS << "function name = " << sanitize(C_.Name) << "_state_name(code)\n";
    OS << "  if false\n";
    for (auto &P : L_.StateCode) {
      OS << "  elseif code == " << P.second << "\n";
      OS << "    name = '" << P.first << "';\n";
    }
    OS << "  else\n";
    OS << "    name = 'none';\n";
    OS << "  end\n";
    OS << "end\n";
  }
};

} // namespace

std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag) {
  LoweringOptions Opts;
  Emitter E(C, Diag, Opts);
  return E.emit();
}

std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag,
                                                 const LoweringOptions &Opts) {
  Emitter E(C, Diag, Opts);
  return E.emit();
}

} // namespace matlab::statechart
