#include "matlab/Flowchart/MflowLinkSim.h"

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>
#include <string>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// MflowLinkSim — the Tier-C continuous-time simulation interpreter.
//
// One in-process pass per major step:
//   1. Resolve every block's input from `Out_` (the previous-step
//      outputs for loop-breakers; this step's outputs for everything
//      already evaluated, since `M_.ExecOrder` is topo-sorted with
//      loop-breakers' outgoing edges dropped).
//   2. Call its evaluator, writing `Out_` and, if a state derivative
//      buffer is being filled, the per-block dx/dt slice.
//   3. The integrator (classic RK4) calls evalAll four times to
//      collect k1..k4, advances Y_ by Σ wi·ki·h.
//
// Inputs are assumed scalar `double` throughout Tier C — sufficient
// for the lowpass / pid_tracking demos. Vector signals + bus types
// are Tier H.
//===----------------------------------------------------------------------===//

namespace matlab::flowchart {

namespace {

// "1, 2, 1" → {1, 2, 1}. Trims whitespace; silently drops malformed
// tokens (the loader has already validated structural shape).
std::vector<double> parsePoly(const std::string &S) {
  std::vector<double> Out;
  std::stringstream SS(S);
  std::string Tok;
  while (std::getline(SS, Tok, ',')) {
    size_t A = Tok.find_first_not_of(" \t");
    if (A == std::string::npos) continue;
    size_t B = Tok.find_last_not_of(" \t");
    try {
      Out.push_back(std::stod(Tok.substr(A, B - A + 1)));
    } catch (...) {
    }
  }
  return Out;
}

const std::string *paramS(const MflBlock &B, const char *Key) {
  auto It = B.Params.find(Key);
  return It == B.Params.end() ? nullptr : &It->second;
}

} // namespace

//===----------------------------------------------------------------------===//
// Tier-H carve-out — `signal_matlab_fcn` expression tree + evaluator.
//
// A tiny recursive-descent parser handles enough of the MATLAB
// expression grammar to express ~80% of "MATLAB Function Block"
// formulas:
//
//   expr  := term (('+'|'-') term)*
//   term  := unary (('*'|'/'|'.*'|'./') unary)*
//   unary := '-' unary | power
//   power := atom ('^' unary)?     ;; right-associative
//   atom  := number | identifier | identifier '(' args ')' | '(' expr ')'
//
// Variables: `u` (alias for `u1`), `u1`..`uN`, `t`, `pi`, `e`.
// Functions: sin/cos/tan/asin/acos/atan/atan2, sinh/cosh/tanh,
//            exp/log/log10/log2/sqrt, abs/sign/floor/ceil/round,
//            min/max/mod/rem/pow/hypot.
//
// The node is a tagged union allocated through `std::unique_ptr` so
// the cache can be heterogeneous without a std::variant. Evaluation
// walks the tree with a small switch.
//===----------------------------------------------------------------------===//

namespace {

using Node = MatlabFcnTree;
using NodeKind = Node::K;

class ExprParser {
public:
  ExprParser(std::string_view S) : S_(S) {}

  std::unique_ptr<Node> parse(std::string &Err) {
    auto E = parseExpr();
    skipWs();
    if (!E || Failed_) {
      Err = ErrMsg_.empty() ? std::string("invalid expression") : ErrMsg_;
      return nullptr;
    }
    if (Pos_ != S_.size()) {
      Err = "trailing content at \"" + std::string(S_.substr(Pos_)) + "\"";
      return nullptr;
    }
    return E;
  }

private:
  std::string_view S_;
  size_t Pos_ = 0;
  bool Failed_ = false;
  std::string ErrMsg_;

  void fail(std::string Msg) {
    if (Failed_) return;
    Failed_ = true;
    ErrMsg_ = std::move(Msg);
  }
  void skipWs() {
    while (Pos_ < S_.size() &&
           (S_[Pos_] == ' ' || S_[Pos_] == '\t' ||
            S_[Pos_] == '\n' || S_[Pos_] == '\r'))
      ++Pos_;
  }
  bool peek(char C) {
    skipWs();
    return Pos_ < S_.size() && S_[Pos_] == C;
  }
  bool consume(char C) {
    skipWs();
    if (Pos_ < S_.size() && S_[Pos_] == C) { ++Pos_; return true; }
    return false;
  }
  bool consume2(char A, char B) {
    skipWs();
    if (Pos_ + 1 < S_.size() && S_[Pos_] == A && S_[Pos_ + 1] == B) {
      Pos_ += 2;
      return true;
    }
    return false;
  }
  std::unique_ptr<Node> num(double V) {
    auto N = std::make_unique<Node>();
    N->Kind = NodeKind::Num;
    N->Num = V;
    return N;
  }
  std::unique_ptr<Node> bin(char Op, std::unique_ptr<Node> L,
                            std::unique_ptr<Node> R) {
    auto N = std::make_unique<Node>();
    N->Kind = NodeKind::Bin;
    N->Op = Op;
    N->Children.push_back(std::move(L));
    N->Children.push_back(std::move(R));
    return N;
  }

  std::unique_ptr<Node> parseExpr() {
    auto L = parseTerm();
    while (L && !Failed_) {
      skipWs();
      if (peek('+'))      { ++Pos_; auto R = parseTerm(); if (!R) return nullptr; L = bin('+', std::move(L), std::move(R)); }
      else if (peek('-')) { ++Pos_; auto R = parseTerm(); if (!R) return nullptr; L = bin('-', std::move(L), std::move(R)); }
      else break;
    }
    return L;
  }
  std::unique_ptr<Node> parseTerm() {
    auto L = parseUnary();
    while (L && !Failed_) {
      skipWs();
      if (consume2('.', '*'))      { auto R = parseUnary(); if (!R) return nullptr; L = bin('*', std::move(L), std::move(R)); }
      else if (consume2('.', '/')) { auto R = parseUnary(); if (!R) return nullptr; L = bin('/', std::move(L), std::move(R)); }
      else if (peek('*'))          { ++Pos_; auto R = parseUnary(); if (!R) return nullptr; L = bin('*', std::move(L), std::move(R)); }
      else if (peek('/'))          { ++Pos_; auto R = parseUnary(); if (!R) return nullptr; L = bin('/', std::move(L), std::move(R)); }
      else break;
    }
    return L;
  }
  std::unique_ptr<Node> parseUnary() {
    skipWs();
    if (peek('-')) {
      ++Pos_;
      auto C = parseUnary();
      if (!C) return nullptr;
      auto N = std::make_unique<Node>();
      N->Kind = NodeKind::Neg;
      N->Children.push_back(std::move(C));
      return N;
    }
    if (peek('+')) { ++Pos_; return parseUnary(); }
    return parsePower();
  }
  std::unique_ptr<Node> parsePower() {
    auto L = parseAtom();
    if (!L) return nullptr;
    skipWs();
    if (consume2('.', '^') || peek('^')) {
      if (Pos_ < S_.size() && S_[Pos_] == '^') ++Pos_;
      auto R = parseUnary();
      if (!R) return nullptr;
      L = bin('^', std::move(L), std::move(R));
    }
    return L;
  }
  std::unique_ptr<Node> parseAtom() {
    skipWs();
    if (Pos_ >= S_.size()) { fail("unexpected end of expression"); return nullptr; }
    char C = S_[Pos_];
    if (C == '(') {
      ++Pos_;
      auto E = parseExpr();
      if (!E) return nullptr;
      if (!consume(')')) { fail("expected ')'"); return nullptr; }
      return E;
    }
    if (std::isdigit(static_cast<unsigned char>(C)) || C == '.') {
      size_t Start = Pos_;
      while (Pos_ < S_.size() &&
             (std::isdigit(static_cast<unsigned char>(S_[Pos_])) ||
              S_[Pos_] == '.')) ++Pos_;
      if (Pos_ < S_.size() && (S_[Pos_] == 'e' || S_[Pos_] == 'E')) {
        ++Pos_;
        if (Pos_ < S_.size() && (S_[Pos_] == '+' || S_[Pos_] == '-')) ++Pos_;
        while (Pos_ < S_.size() &&
               std::isdigit(static_cast<unsigned char>(S_[Pos_]))) ++Pos_;
      }
      try { return num(std::stod(std::string(S_.substr(Start, Pos_ - Start)))); }
      catch (...) { fail("invalid number literal"); return nullptr; }
    }
    if (std::isalpha(static_cast<unsigned char>(C)) || C == '_') {
      size_t Start = Pos_;
      while (Pos_ < S_.size() &&
             (std::isalnum(static_cast<unsigned char>(S_[Pos_])) ||
              S_[Pos_] == '_')) ++Pos_;
      std::string Name(S_.substr(Start, Pos_ - Start));
      skipWs();
      if (peek('(')) {
        ++Pos_;
        auto N = std::make_unique<Node>();
        N->Kind = NodeKind::Call;
        N->Name = Name;
        if (!peek(')')) {
          while (true) {
            auto Arg = parseExpr();
            if (!Arg) return nullptr;
            N->Children.push_back(std::move(Arg));
            if (consume(',')) continue;
            break;
          }
        }
        if (!consume(')')) { fail("expected ')' in call to " + Name); return nullptr; }
        return N;
      }
      auto N = std::make_unique<Node>();
      N->Kind = NodeKind::Var;
      N->Name = std::move(Name);
      return N;
    }
    fail(std::string("unexpected character '") + C + "'");
    return nullptr;
  }
};

double evalMatlabFcn(const Node *N,
                     const std::vector<double> &U,
                     double T) {
  if (!N) return 0.0;
  switch (N->Kind) {
  case NodeKind::Num: return N->Num;
  case NodeKind::Var: {
    const std::string &Name = N->Name;
    if (Name == "t")   return T;
    if (Name == "u")   return U.empty() ? 0.0 : U[0];
    if (Name == "pi")  return M_PI;
    if (Name == "e")   return M_E;
    if (Name.size() > 1 && Name[0] == 'u' &&
        std::all_of(Name.begin() + 1, Name.end(),
                    [](char C) {
                      return std::isdigit(static_cast<unsigned char>(C));
                    })) {
      int Idx = std::stoi(Name.substr(1));
      if (Idx >= 1 && static_cast<size_t>(Idx) <= U.size())
        return U[Idx - 1];
    }
    return 0.0;
  }
  case NodeKind::Neg:
    return -evalMatlabFcn(N->Children[0].get(), U, T);
  case NodeKind::Bin: {
    double L = evalMatlabFcn(N->Children[0].get(), U, T);
    double R = evalMatlabFcn(N->Children[1].get(), U, T);
    switch (N->Op) {
    case '+': return L + R;
    case '-': return L - R;
    case '*': return L * R;
    case '/': return L / R;
    case '^': return std::pow(L, R);
    }
    return 0.0;
  }
  case NodeKind::Call: {
    const std::string &F = N->Name;
    auto arg = [&](size_t I) {
      return I < N->Children.size()
                 ? evalMatlabFcn(N->Children[I].get(), U, T)
                 : 0.0;
    };
    if (F == "sin")    return std::sin(arg(0));
    if (F == "cos")    return std::cos(arg(0));
    if (F == "tan")    return std::tan(arg(0));
    if (F == "asin")   return std::asin(arg(0));
    if (F == "acos")   return std::acos(arg(0));
    if (F == "atan")   return std::atan(arg(0));
    if (F == "atan2")  return std::atan2(arg(0), arg(1));
    if (F == "sinh")   return std::sinh(arg(0));
    if (F == "cosh")   return std::cosh(arg(0));
    if (F == "tanh")   return std::tanh(arg(0));
    if (F == "exp")    return std::exp(arg(0));
    if (F == "log")    return std::log(arg(0));
    if (F == "log10")  return std::log10(arg(0));
    if (F == "log2")   return std::log2(arg(0));
    if (F == "sqrt")   return std::sqrt(arg(0));
    if (F == "abs")    return std::fabs(arg(0));
    if (F == "sign") { double V = arg(0); return (V > 0) - (V < 0); }
    if (F == "floor")  return std::floor(arg(0));
    if (F == "ceil")   return std::ceil(arg(0));
    if (F == "round")  return std::round(arg(0));
    if (F == "min")    return std::fmin(arg(0), arg(1));
    if (F == "max")    return std::fmax(arg(0), arg(1));
    if (F == "mod")    return std::fmod(arg(0), arg(1));
    if (F == "rem")    return arg(0) - arg(1) * std::trunc(arg(0) / arg(1));
    if (F == "pow")    return std::pow(arg(0), arg(1));
    if (F == "hypot")  return std::hypot(arg(0), arg(1));
    if (F == "square") { double V = arg(0); return V * V; }
    return 0.0;
  }
  }
  return 0.0;
}

} // namespace

// Item-4 — forward declarations for the function-body parser +
// interpreter. Bodies live further down in this file (after the
// expression evaluator) but the constructor and evaluator branch
// above use them.
namespace {
std::pair<std::unique_ptr<MflowLinkSim::MatlabFunctionState>, std::string>
parseMatlabFunctionBody(const std::string &Source);
double runMatlabFunction(const MflowLinkSim::MatlabFunctionState &S,
                         const std::vector<double> &Inputs, double T);
} // namespace

double MflowLinkSim::paramD(const MflBlock &B, const char *Key, double Def) {
  auto It = B.Params.find(Key);
  if (It == B.Params.end()) return Def;
  try {
    return std::stod(It->second);
  } catch (...) {
    return Def;
  }
}

//===----------------------------------------------------------------------===//
// Construction — pre-compute everything per-block we'd otherwise
// reparse on every step.
//===----------------------------------------------------------------------===//

MflowLinkSim::MflowLinkSim(const MflowLinkModel &M) : M_(M) {
  const size_t N = M_.Blocks.size();
  Out_.assign(N, 0.0);
  OutWidth_.resize(N, 1);
  VecOut_.assign(N, {});
  for (size_t I = 0; I < N; ++I) {
    int W = M_.Blocks[I].OutWidth;
    if (W < 1) W = 1;
    OutWidth_[I] = W;
    if (W > 1) VecOut_[I].assign(W, 0.0);
  }
  Inputs_.assign(N, {});
  StateOffset_.assign(N, 0);
  DiscStateOffset_.assign(N, 0);
  NextFire_.assign(N, std::numeric_limits<double>::infinity());
  TFCache_.assign(N, {});
  Gate_.assign(N, -1);
  GateEdge_.assign(N, false);
  PrevOut_.assign(N, 0.0);

  std::unordered_map<std::string, size_t> IdxOf;
  for (size_t I = 0; I < N; ++I) IdxOf[M_.Blocks[I].Id] = I;

  // Resolve EnableSource → block index. Lowering has already
  // validated the reference, so any failure here is a logic error.
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    if (B.EnableSource.empty()) continue;
    auto It = IdxOf.find(B.EnableSource);
    if (It != IdxOf.end()) Gate_[I] = static_cast<int>(It->second);
    GateEdge_[I] = B.EnableEdgeTriggered;
  }

  // Per-block continuous-state offset.
  size_t Off = 0;
  for (size_t I = 0; I < N; ++I) {
    StateOffset_[I] = Off;
    Off += static_cast<size_t>(M_.Blocks[I].ContStateCount);
  }
  Y_.assign(M_.ContStateCount, 0.0);

  // Per-block discrete-state offset (one slot for Unit Delay / ZOH).
  size_t DOff = 0;
  for (size_t I = 0; I < N; ++I) {
    DiscStateOffset_[I] = DOff;
    DOff += static_cast<size_t>(M_.Blocks[I].DiscStateCount);
  }
  Z_.assign(M_.DiscStateCount, 0.0);
  Znext_.assign(M_.DiscStateCount, 0.0);
  DiscPrevU_.assign(N, 0.0);

  // Input wiring. Sum / Product blocks read in1, in2, …; every other
  // single-input block reads "in".
  for (auto &E : M_.Edges) {
    auto FI = IdxOf.find(E.FromBlock);
    auto TI = IdxOf.find(E.ToBlock);
    if (FI == IdxOf.end() || TI == IdxOf.end()) continue;
    Inputs_[TI->second].push_back({FI->second, E.ToPort});
  }

  // Cache transfer-function coefficients. For `signal_zero_pole`,
  // we run the same path after expanding zeros/poles into num/den
  // polynomials — the evaluator reuses the transfer_fcn machinery
  // verbatim from then on.
  TransportBuf_.assign(N, {});
  Lookup1DCache_.assign(N, {});
  Lookup2DCache_.assign(N, {});
  NoiseSeed_.assign(N, 0);
  MatlabFcnCache_.resize(N);
  MatlabFnCache_.resize(N);
  auto expandPoly = [](const std::vector<double> &Roots) {
    // Product over `(s - r_k)` expanded to coefficient form, highest
    // power first. Empty roots ⇒ {1} (the "constant 1" polynomial).
    std::vector<double> P{1.0};
    for (double R : Roots) {
      std::vector<double> Q(P.size() + 1, 0.0);
      for (size_t I = 0; I < P.size(); ++I) {
        Q[I]     += P[I];
        Q[I + 1] += -R * P[I];
      }
      P = std::move(Q);
    }
    return P;
  };
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    if (B.Kind == "signal_transfer_fcn" ||
        B.Kind == "signal_discrete_filter") {
      auto *NumS = paramS(B, "num");
      auto *DenS = paramS(B, "den");
      TFCache_[I].Num = parsePoly(NumS ? *NumS : "1");
      TFCache_[I].Den = parsePoly(DenS ? *DenS : "1");
      TFCache_[I].Valid = !TFCache_[I].Den.empty();
    } else if (B.Kind == "signal_zero_pole") {
      // Param shape: `zeros` and `poles` are comma-separated real
      // roots; `gain` is a scalar leading coefficient. Complex root
      // pairs aren't accepted in the Tier-H pass — would need a
      // complex-poly parser. Pure-real zero-pole models cover the
      // common pole-placement / PID-with-zeros cases.
      auto *ZS = paramS(B, "zeros");
      auto *PS = paramS(B, "poles");
      double Gain = MflowLinkSim::paramD(B, "gain", 1.0);
      auto Zeros = parsePoly(ZS ? *ZS : "");
      auto Poles = parsePoly(PS ? *PS : "");
      auto Num = expandPoly(Zeros);
      auto Den = expandPoly(Poles);
      for (auto &C : Num) C *= Gain;
      TFCache_[I].Num = std::move(Num);
      TFCache_[I].Den = std::move(Den);
      TFCache_[I].Valid = !TFCache_[I].Den.empty();
    } else if (B.Kind == "signal_lookup_1d") {
      auto *X = paramS(B, "breakpointsX");
      auto *Y = paramS(B, "tableData");
      Lookup1DCache_[I].X = parsePoly(X ? *X : "");
      Lookup1DCache_[I].Y = parsePoly(Y ? *Y : "");
      // Sort by X (breakpoints must be monotonically increasing —
      // we re-sort defensively in case the user supplied them
      // already-sorted-but-with-noise, paired with their Y values).
      if (Lookup1DCache_[I].X.size() == Lookup1DCache_[I].Y.size() &&
          !Lookup1DCache_[I].X.empty()) {
        std::vector<std::pair<double, double>> Pairs;
        Pairs.reserve(Lookup1DCache_[I].X.size());
        for (size_t K = 0; K < Lookup1DCache_[I].X.size(); ++K)
          Pairs.emplace_back(Lookup1DCache_[I].X[K], Lookup1DCache_[I].Y[K]);
        std::sort(Pairs.begin(), Pairs.end());
        for (size_t K = 0; K < Pairs.size(); ++K) {
          Lookup1DCache_[I].X[K] = Pairs[K].first;
          Lookup1DCache_[I].Y[K] = Pairs[K].second;
        }
        Lookup1DCache_[I].Valid = true;
      }
    } else if (B.Kind == "signal_lookup_2d") {
      auto *X = paramS(B, "breakpointsX");
      auto *Y = paramS(B, "breakpointsY");
      auto *T = paramS(B, "tableData");
      Lookup2DCache_[I].X = parsePoly(X ? *X : "");
      Lookup2DCache_[I].Y = parsePoly(Y ? *Y : "");
      Lookup2DCache_[I].Z = parsePoly(T ? *T : "");
      Lookup2DCache_[I].Valid = !Lookup2DCache_[I].X.empty() &&
                                !Lookup2DCache_[I].Y.empty() &&
                                Lookup2DCache_[I].Z.size() ==
                                  Lookup2DCache_[I].X.size() *
                                  Lookup2DCache_[I].Y.size();
    } else if (B.Kind == "signal_transport_delay") {
      TransportBuf_[I].Delay =
          MflowLinkSim::paramD(B, "delay", 0.0);
      TransportBuf_[I].InitialOutput =
          MflowLinkSim::paramD(B, "initialOutput", 0.0);
    } else if (B.Kind == "signal_matlab_fcn") {
      // Parse `params.expression` once. A failed parse leaves the
      // cache slot empty; the evaluator emits 0.0 and the user has
      // already seen the diagnostic from the validation pass in
      // lowerSignalFlow (driver-side — see SignalFlowLowering.cpp).
      auto *ES = paramS(B, "expression");
      if (ES && !ES->empty()) {
        std::string Err;
        ExprParser P(*ES);
        MatlabFcnCache_[I] = P.parse(Err);
        (void)Err; // already reported at lowering
      }
      // Item-4 — alternative path: full MATLAB function body.
      // Parsed via the matlab_llvm lexer + parser and walked by
      // the AST interpreter in `runMatlabFunction`. Same lowering-
      // time validation contract as the expression path.
      auto *FB = paramS(B, "function_body");
      if (FB && !FB->empty()) {
        auto Pair = parseMatlabFunctionBody(*FB);
        if (Pair.first) MatlabFnCache_[I] = std::move(Pair.first);
        // Parse failure was already reported at lowering — leaving
        // the cache empty makes the evaluator fall through to 0.
      }
    }
  }

  StepSize_ = 0.01;
  if (M_.Solver.MaxStep != "auto") {
    try {
      double H = std::stod(M_.Solver.MaxStep);
      if (H > 0.0) StepSize_ = H;
    } catch (...) {
    }
  }
  // Item-2 — adaptive step is enabled when the model asks for
  // variable_step + an adaptive algorithm. Fixed-step mode keeps
  // the legacy classic RK4 path.
  AdaptiveSolver_ = (M_.Solver.Type != "fixed_step") &&
                    (M_.Solver.Algorithm == "ode45" ||
                     M_.Solver.Algorithm == "ode23");
  CurrentAdaptiveH_ = StepSize_;

  if (M_.Snapshot.Enabled && M_.Snapshot.Depth > 0)
    SnapshotCap_ = static_cast<size_t>(M_.Snapshot.Depth);
  else
    SnapshotCap_ = 0;

  // Log columns: anything with `data.log_signal: true`, plus every
  // scope / to_workspace / display (they implicitly log).
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    bool Implicit = B.Kind == "signal_scope" ||
                    B.Kind == "signal_display" ||
                    B.Kind == "signal_to_workspace";
    if (!B.LogSignal && !Implicit) continue;
    std::string Name = B.Id;
    if (B.Kind == "signal_to_workspace") {
      if (auto *V = paramS(B, "variableName")) Name = *V;
    }
    int W = OutWidth_[I];
    if (W <= 1) {
      LogBlocks_.push_back(I);
      LogElements_.push_back(0);
      LogNames_.push_back(Name);
    } else {
      // Item-1 — one CSV column per element. Naming follows MATLAB
      // indexing (1-based on disk) to match what users see in the
      // IDE's scope.
      for (int E = 0; E < W; ++E) {
        LogBlocks_.push_back(I);
        LogElements_.push_back(E);
        LogNames_.push_back(Name + "[" + std::to_string(E + 1) + "]");
      }
    }
  }
  LogColumns_.assign(LogNames_.size(), {});
}

//===----------------------------------------------------------------------===//
// reset — pull initial conditions out of `params` into Y_, drop logs.
//===----------------------------------------------------------------------===//

void MflowLinkSim::reset() {
  T_ = M_.Solver.StartTime;
  MajorSteps_ = 0;
  std::fill(Y_.begin(), Y_.end(), 0.0);
  std::fill(Z_.begin(), Z_.end(), 0.0);
  std::fill(Znext_.begin(), Znext_.end(), 0.0);
  std::fill(DiscPrevU_.begin(), DiscPrevU_.end(), 0.0);
  std::fill(Out_.begin(), Out_.end(), 0.0);
  std::fill(PrevOut_.begin(), PrevOut_.end(), 0.0);
  for (auto &C : LogColumns_) C.clear();
  Snapshots_.clear();
  LogsTruncated_ = false;
  BlockCursor_ = 0;
  ZCQueue_.clear();

  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const auto &B = M_.Blocks[I];
    if (B.Kind == "signal_integrator") {
      Y_[StateOffset_[I]] = paramD(B, "initialCondition", 0.0);
    } else if (B.Kind == "signal_state_space") {
      // x0 may be a vector literal — Tier C reads scalar only; vector
      // ICs are picked up when state_space gets its full evaluator.
      double X0 = paramD(B, "x0", 0.0);
      for (int J = 0; J < B.ContStateCount; ++J)
        Y_[StateOffset_[I] + J] = X0;
    } else if (B.Kind == "signal_relay") {
      // Initial relay state: `initialState` (bool/0|1) wins, else
      // start in the "off" branch. The on/off VALUES (not the
      // boolean state) reach Out_[I] at first evalAll.
      double IS = paramD(B, "initialState", 0.0);
      Z_[DiscStateOffset_[I]] = IS > 0.5 ? 1.0 : 0.0;
    } else if (B.Kind == "signal_unit_delay" || B.Kind == "signal_zoh") {
      // Latched output starts at the initial-value param (Unit Delay)
      // or 0 (ZOH — held by definition once the first tick fires).
      double IV = paramD(B, "initialValue", 0.0);
      Z_[DiscStateOffset_[I]] = IV;
      Znext_[DiscStateOffset_[I]] = IV;
      // First fire-tick is at startTime + 0 (Simulink convention) —
      // the value latched at t=startTime is what the block outputs
      // through the first sample interval. A future `sampleOffset`
      // param can override this.
      NextFire_[I] = M_.Solver.StartTime;
    } else if (B.Kind == "signal_discrete_integrator" ||
               B.Kind == "signal_rate_transition" ||
               B.Kind == "signal_discrete_filter") {
      // Initial discrete state: `initialCondition` (or 0). For
      // signal_discrete_filter the IIR taps all start at 0; the
      // first sample at NextFire_ then writes the new state.
      double IC = paramD(B, "initialCondition", 0.0);
      for (int J = 0; J < B.DiscStateCount; ++J)
        Z_[DiscStateOffset_[I] + J] = J == 0 ? IC : 0.0;
      // First fire-tick = startTime + period. The state already
      // carries the initial condition, so the integration step
      // y[1] = y[0] + h·u happens at t = startTime + period —
      // matches Simulink's "Integrator: update after step" model.
      // Unit Delay / ZOH differ: they need to latch a sample *at*
      // startTime to have any output during the first interval.
      double Period =
          B.SamplePeriod > 0.0 ? B.SamplePeriod : 1.0;
      NextFire_[I] = M_.Solver.StartTime + Period;
    } else if (B.Kind == "signal_transport_delay") {
      TransportBuf_[I].Samples.clear();
      // Prime the history with a single sample at startTime so
      // the first major step's interpolation has something to
      // read.
      TransportBuf_[I].Samples.push_back(
          {M_.Solver.StartTime, TransportBuf_[I].InitialOutput});
    } else if (B.Kind == "signal_noise") {
      // Per-block xorshift seed. The default seed makes the same
      // model reproducible across runs; users can override via
      // `params.seed`.
      uint64_t Seed =
          static_cast<uint64_t>(paramD(B, "seed", 1.0));
      if (Seed == 0) Seed = 0xC0FFEE12345678ABULL;
      NoiseSeed_[I] = Seed;
    }
  }

  // Run one evaluation at t=startTime so the very first logged sample
  // reflects t=0 outputs, not the post-construction zeros.
  evalAll(T_, Y_.data(), nullptr);
  // Relay's initial output depends on its input at t=0 — if the
  // input is already past either threshold, the latched bit needs
  // to reflect that before the first logSample.
  commitRelayState();
  evalAll(T_, Y_.data(), nullptr);
  // Snapshot the initial outputs as the "previous" baseline so the
  // first major step's edge-trigger detection compares against the
  // post-init values, not the all-zero default. Without this, an
  // already-high gate signal at t=0 would falsely "rise" during
  // the first step.
  PrevOut_ = Out_;

  // Seed zero-crossing signs from the initial outputs so the first
  // major step can detect a flip cleanly.
  ZCSign_.assign(M_.ZeroCrossings.size(), 0);
  for (size_t K = 0; K < M_.ZeroCrossings.size(); ++K)
    ZCSign_[K] = predicateSign(K);

  logSample();
}

//===----------------------------------------------------------------------===//
// evalAll — one pass over ExecOrder, writes Out_ and (if requested)
//           the dx/dt buffer.
//===----------------------------------------------------------------------===//

void MflowLinkSim::evalAll(double T, const double *State, double *Deriv) {
  if (Deriv) std::memset(Deriv, 0, sizeof(double) * Y_.size());

  auto inputOf = [&](size_t I, const char *PortId) -> double {
    for (auto &P : Inputs_[I])
      if (P.DstPort == PortId) return Out_[P.SrcBlock];
    return 0.0;
  };
  auto sumInput = [&](size_t I, const std::string &Port) -> double {
    // Multiple edges may land on the same input port — sum them
    // (matches Simulink's implicit summing convention for vector
    // joins, and harmless for the single-edge common case).
    double V = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == Port) V += Out_[P.SrcBlock];
    return V;
  };

  for (size_t Pos = 0; Pos < M_.ExecOrder.size(); ++Pos) {
    size_t I = M_.ExecOrder[Pos];
    const auto &B = M_.Blocks[I];
    const std::string &K = B.Kind;

    // Tier F — conditional-subsystem gate. The gate signal is read
    // from `Out_` which has already been written for every earlier
    // block in topo order; the gate source therefore appears
    // before any gated block (an enable signal can't be downstream
    // of the gate, which the lowering ensures by edge-rewiring
    // through the subsystem boundary). A non-positive gate skips
    // the evaluator entirely (output and discrete state hold) and
    // zeros the continuous-state derivative slice so any RK4 sub-
    // step sees `dx/dt = 0` for the gated block.
    //
    // Edge-triggered gate (Tier F carve-out — from a
    // `signal_triggered_subsystem`): the block fires only when the
    // gate signal has just *risen* through zero between the previous
    // and current major step. `PrevOut_` captures the last logged
    // value of every block; on a rising edge it's ≤ 0 and the
    // current `Out_` is > 0, so the gate opens for exactly one
    // step.
    if (Gate_[I] >= 0) {
      bool Open;
      if (GateEdge_[I]) {
        double Now = Out_[Gate_[I]];
        double Before = PrevOut_[Gate_[I]];
        Open = (Before <= 0.0 && Now > 0.0);
      } else {
        Open = Out_[Gate_[I]] > 0.0;
      }
      if (!Open) {
        // Level-gated (`signal_enabled_subsystem`): hold the prior
        // output and freeze any continuous state — Simulink's
        // "Output when disabled: held".
        // Edge-triggered (`signal_triggered_subsystem`): drive the
        // output to zero outside the trigger window so a downstream
        // accumulator/integrator only sees the value during the
        // single firing step — Simulink's "Output when disabled:
        // reset". This is what makes a function-call-generator +
        // triggered-pass + integrator add up to one count per
        // pulse instead of running away.
        if (GateEdge_[I]) Out_[I] = 0.0;
        if (Deriv && B.ContStateCount > 0) {
          size_t Off = StateOffset_[I];
          for (int J = 0; J < B.ContStateCount; ++J) Deriv[Off + J] = 0.0;
        }
        continue;
      }
    }

    if (K == "signal_constant") {
      // Item-1 — vector-literal value (`"[1 2 3]"`). Scalar still
      // hits Out_[I]; vectors fill VecOut_[I] and mirror the first
      // element into Out_[I] for scalar consumers.
      if (OutWidth_[I] > 1) {
        const std::string *V = paramS(B, "value");
        std::string S = V ? *V : "0";
        auto A = S.find('[');
        auto C0 = S.rfind(']');
        if (A != std::string::npos && C0 != std::string::npos && A < C0)
          S = S.substr(A + 1, C0 - A - 1);
        std::vector<double> Vals;
        std::string Tok;
        for (size_t Ki = 0; Ki <= S.size(); ++Ki) {
          char Cc = Ki < S.size() ? S[Ki] : ',';
          if (Cc == ',' || Cc == ' ' || Cc == '\t' || Cc == ';') {
            if (!Tok.empty()) {
              try { Vals.push_back(std::stod(Tok)); } catch (...) {}
              Tok.clear();
            }
          } else {
            Tok.push_back(Cc);
          }
        }
        Vals.resize(OutWidth_[I], 0.0);
        VecOut_[I] = Vals;
        Out_[I] = Vals.front();
      } else {
        Out_[I] = paramD(B, "value", 0.0);
      }
    } else if (K == "signal_step") {
      double ST = paramD(B, "stepTime", 1.0);
      double IV = paramD(B, "initialValue", 0.0);
      double FV = paramD(B, "finalValue", 1.0);
      Out_[I] = (T >= ST) ? FV : IV;
    } else if (K == "signal_sine") {
      double A  = paramD(B, "amplitude", 1.0);
      double Bs = paramD(B, "bias", 0.0);
      double F  = paramD(B, "frequency", 1.0);
      double P  = paramD(B, "phase", 0.0);
      Out_[I] = A * std::sin(F * T + P) + Bs;
    } else if (K == "signal_ramp") {
      double Slope = paramD(B, "slope", 1.0);
      double Start = paramD(B, "startTime", 0.0);
      double Init  = paramD(B, "initialOutput", 0.0);
      Out_[I] = T >= Start ? Init + Slope * (T - Start) : Init;
    } else if (K == "signal_pulse") {
      double A  = paramD(B, "amplitude", 1.0);
      double Pp = paramD(B, "period", 1.0);
      double W  = paramD(B, "pulseWidth", 50.0);   // percent
      double D  = paramD(B, "phaseDelay", 0.0);
      double Tt = std::fmod(T - D, Pp);
      if (Tt < 0.0) Tt += Pp;
      Out_[I] = (Tt < Pp * W * 0.01) ? A : 0.0;
    } else if (K == "signal_function_call_generator") {
      // Tier F carve-out — emit `1` for one major step at the
      // start of every `period`; `0` otherwise. With a configured
      // period P and current time T, the pulse fires when
      // `t mod P` is within one step of zero. This is what drives
      // a `signal_triggered_subsystem` to fire — the rising edge
      // of this output is the trigger event.
      double Pp = paramD(B, "period", 1.0);
      if (Pp <= 0.0) Pp = 1.0;
      double Phase = paramD(B, "phaseDelay", 0.0);
      double Tt = std::fmod(T - Phase, Pp);
      if (Tt < 0.0) Tt += Pp;
      // Width is 1.5 × the integration step. Half a step at the
      // *trailing* edge would miss the period boundary on rounding-
      // up drift; 1.5 × means even a noticeable accumulation of
      // float error (T_ a few ULP past N·P) lands inside the
      // window. The rising-edge detector keys on PrevOut_[clk] vs
      // Out_[clk], so a 2-step-wide pulse still fires the trigger
      // exactly once per period.
      double Width = StepSize_ * 1.5;
      Out_[I] = (Tt < Width || Tt > Pp - 1e-12) ? 1.0 : 0.0;
    } else if (K == "signal_gain") {
      double G = paramD(B, "gain", 1.0);
      if (OutWidth_[I] > 1) {
        // Find the upstream block driving "in" and pull its vector
        // slice; broadcast scalars by repetition.
        size_t Src = static_cast<size_t>(-1);
        for (auto &P : Inputs_[I])
          if (P.DstPort == "in") { Src = P.SrcBlock; break; }
        VecOut_[I].assign(OutWidth_[I], 0.0);
        if (Src != static_cast<size_t>(-1)) {
          if (OutWidth_[Src] == 1) {
            double V = G * Out_[Src];
            std::fill(VecOut_[I].begin(), VecOut_[I].end(), V);
          } else {
            const auto &SV = VecOut_[Src];
            for (int E = 0; E < OutWidth_[I]; ++E)
              VecOut_[I][E] = G * (E < (int)SV.size() ? SV[E] : 0.0);
          }
        }
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = G * inputOf(I, "in");
      }
    } else if (K == "signal_abs") {
      Out_[I] = std::fabs(inputOf(I, "in"));
    } else if (K == "signal_saturation") {
      double Lo = paramD(B, "lowerLimit", -1.0);
      double Hi = paramD(B, "upperLimit",  1.0);
      double U  = inputOf(I, "in");
      Out_[I] = U < Lo ? Lo : (U > Hi ? Hi : U);
    } else if (K == "signal_sum") {
      if (OutWidth_[I] > 1) {
        // Element-wise vector sum. Each input port contributes one
        // slice; scalar sources broadcast.
        auto *Signs = paramS(B, "signs");
        VecOut_[I].assign(OutWidth_[I], 0.0);
        size_t N = Signs ? Signs->size() : Inputs_[I].size();
        for (size_t Ki = 0; Ki < N; ++Ki) {
          char Sg = (Signs && Ki < Signs->size()) ? (*Signs)[Ki] : '+';
          std::string Port = "in" + std::to_string(Ki + 1);
          for (auto &P : Inputs_[I]) {
            if (P.DstPort != Port && !(Ki == 0 && P.DstPort == "in")) continue;
            int SW = OutWidth_[P.SrcBlock];
            for (int E = 0; E < OutWidth_[I]; ++E) {
              double V = SW == 1 ? Out_[P.SrcBlock]
                                 : (E < (int)VecOut_[P.SrcBlock].size()
                                        ? VecOut_[P.SrcBlock][E] : 0.0);
              VecOut_[I][E] += (Sg == '-') ? -V : V;
            }
          }
        }
        Out_[I] = VecOut_[I].front();
      } else {
        double Sum = 0.0;
        auto *Signs = paramS(B, "signs");
        if (Signs && !Signs->empty()) {
          for (size_t Ki = 0; Ki < Signs->size(); ++Ki) {
            char Sg = (*Signs)[Ki];
            double V = sumInput(I, "in" + std::to_string(Ki + 1));
            Sum += (Sg == '-') ? -V : V;
          }
        } else {
          // No `signs` declared — sum every connected input port.
          for (auto &P : Inputs_[I]) Sum += Out_[P.SrcBlock];
        }
        Out_[I] = Sum;
      }
    } else if (K == "signal_product") {
      int NIn = static_cast<int>(paramD(B, "numInputs", 2.0));
      if (NIn < 1) NIn = 1;
      double Prod = 1.0;
      for (int Ki = 1; Ki <= NIn; ++Ki)
        Prod *= sumInput(I, "in" + std::to_string(Ki));
      Out_[I] = Prod;
    } else if (K == "signal_integrator") {
      size_t Off = StateOffset_[I];
      Out_[I] = State[Off];
      if (Deriv) Deriv[Off] = inputOf(I, "in");
    } else if (K == "signal_transfer_fcn") {
      const auto &TF = TFCache_[I];
      int N = static_cast<int>(TF.Den.size()) - 1;
      double U = inputOf(I, "in");
      if (N <= 0 || !TF.Valid) {
        // Static gain: y = (num[0]/den[0]) * u.
        double NumLead = TF.Num.empty() ? 0.0 : TF.Num.front();
        double DenLead = TF.Den.empty() ? 1.0 : TF.Den.front();
        Out_[I] = (NumLead / DenLead) * U;
      } else {
        // Controllable canonical form, den = a_n s^n + ... + a_0,
        // num = b_m s^m + ... + b_0 (right-padded to length n+1 with
        // leading zeros). State x_0..x_{n-1}; x'_k = x_{k+1} for k<n-1
        // and x'_{n-1} = (u - Σ a_k/a_n · x_k) / 1 (already normalised).
        size_t Off = StateOffset_[I];
        double Lead = TF.Den.front();
        if (Deriv) {
          for (int Ki = 0; Ki < N - 1; ++Ki)
            Deriv[Off + Ki] = State[Off + Ki + 1];
          double Last = U / Lead;
          // a_k coefficient → Den[N-k] / Lead.
          for (int Ki = 0; Ki < N; ++Ki)
            Last -= (TF.Den[N - Ki] / Lead) * State[Off + Ki];
          Deriv[Off + N - 1] = Last;
        }
        // Output y = Σ b_k · x_k for strictly proper TF. For
        // not-strictly-proper, the b_n direct-feedthrough term would
        // add (b_n / a_n) · u — but we marked those as non-loop-
        // breakers in lowering, and the topo sort already handles the
        // direct-feedthrough wiring. For Tier C we restrict to the
        // strictly-proper case (degNum < degDen); the loader / lower
        // already classified that as the loop-breaker path.
        std::vector<double> NPad(N + 1, 0.0);
        int Mdeg = static_cast<int>(TF.Num.size()) - 1;
        for (int Ki = 0; Ki <= Mdeg; ++Ki)
          NPad[N - Mdeg + Ki] = TF.Num[Ki];
        double Y = 0.0;
        for (int Ki = 0; Ki < N; ++Ki)
          // Item-3 — correct controllable-canonical-form output:
          // C = num coefficients (NO division by Lead). The state
          // x is already scaled because B = 1/Lead, so y = C·x
          // gives the right gain. The previous `/Lead` was a bug
          // masked by every demo using `den = "1, ..."` (Lead = 1).
          Y += NPad[N - Ki] * State[Off + Ki];
        Out_[I] = Y;
      }
    } else if (K == "signal_state_space") {
      // Tier-C: only the scalar SISO case (n = ContStateCount, single
      // input/output) with D = 0 (the lowering already marked D ≠ 0
      // as a non-loop-breaker, which we don't support yet).
      auto parseMatrix = [](const std::string &S, std::vector<double> &Vals,
                            int &Rows, int &Cols) {
        Vals.clear();
        Rows = 0;
        Cols = 0;
        std::string T = S;
        auto F = T.find('[');
        if (F != std::string::npos) T.erase(0, F + 1);
        auto L = T.rfind(']');
        if (L != std::string::npos) T.erase(L);
        std::stringstream RS(T);
        std::string Row;
        while (std::getline(RS, Row, ';')) {
          int C = 0;
          std::stringstream CS(Row);
          std::string Tok;
          while (CS >> Tok) {
            try {
              Vals.push_back(std::stod(Tok));
            } catch (...) {
            }
            ++C;
          }
          if (C > 0) {
            ++Rows;
            Cols = C;
          }
        }
      };
      const std::string *AS = paramS(B, "A");
      const std::string *BS = paramS(B, "B");
      const std::string *CS = paramS(B, "C");
      std::vector<double> A, Bm, Cm;
      int Ar = 0, Ac = 0, Br = 0, Bc = 0, Cr = 0, Cc = 0;
      if (AS) parseMatrix(*AS, A, Ar, Ac);
      if (BS) parseMatrix(*BS, Bm, Br, Bc);
      if (CS) parseMatrix(*CS, Cm, Cr, Cc);
      int n = B.ContStateCount;
      size_t Off = StateOffset_[I];
      double U = inputOf(I, "in");
      if (Deriv && static_cast<int>(A.size()) == n * n &&
          static_cast<int>(Bm.size()) >= n) {
        for (int Ri = 0; Ri < n; ++Ri) {
          double D = Bm[Ri] * U;
          for (int Ci = 0; Ci < n; ++Ci)
            D += A[Ri * n + Ci] * State[Off + Ci];
          Deriv[Off + Ri] = D;
        }
      }
      double Y = 0.0;
      if (static_cast<int>(Cm.size()) >= n)
        for (int Ki = 0; Ki < n; ++Ki) Y += Cm[Ki] * State[Off + Ki];
      Out_[I] = Y;
    } else if (K == "signal_scope" || K == "signal_display" ||
               K == "signal_to_workspace" || K == "signal_terminator") {
      if (OutWidth_[I] > 1) {
        // Item-1 — mirror the vector input verbatim so the scope's
        // per-element log columns capture every element of the
        // upstream signal, not just the first.
        size_t Src = static_cast<size_t>(-1);
        for (auto &P : Inputs_[I])
          if (P.DstPort == "in") { Src = P.SrcBlock; break; }
        VecOut_[I].assign(OutWidth_[I], 0.0);
        if (Src != static_cast<size_t>(-1)) {
          int SW = OutWidth_[Src];
          if (SW == 1) {
            std::fill(VecOut_[I].begin(), VecOut_[I].end(), Out_[Src]);
          } else {
            for (int E = 0; E < OutWidth_[I]; ++E)
              VecOut_[I][E] = E < (int)VecOut_[Src].size()
                                 ? VecOut_[Src][E] : 0.0;
          }
        }
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = inputOf(I, "in");
      }
    } else if (K == "signal_mux") {
      // Item-1 — concatenate input slices into a single vector
      // output. Scalar inputs contribute one element each; vector
      // inputs contribute their full width. The width-inference
      // pass already sized OutWidth_[I] = sum of input widths.
      if (OutWidth_[I] > 1) {
        VecOut_[I].assign(OutWidth_[I], 0.0);
        size_t Pos = 0;
        // Visit input ports in port-id order so the concatenation
        // is deterministic. Sum-style "in1", "in2", … and unnamed
        // single-input wires are both handled.
        std::vector<std::pair<int, size_t>> Order;
        for (size_t Idx = 0; Idx < Inputs_[I].size(); ++Idx) {
          const auto &P = Inputs_[I][Idx];
          int N = 1;
          if (P.DstPort.size() > 1 && P.DstPort[0] == 'i' &&
              P.DstPort[1] == 'n') {
            try { N = std::stoi(P.DstPort.substr(2)); } catch (...) { N = 1; }
          }
          Order.emplace_back(N, Idx);
        }
        std::sort(Order.begin(), Order.end());
        for (auto &OEnt : Order) {
          const auto &P = Inputs_[I][OEnt.second];
          int SW = OutWidth_[P.SrcBlock];
          if (SW == 1) {
            if (Pos < VecOut_[I].size())
              VecOut_[I][Pos++] = Out_[P.SrcBlock];
          } else {
            for (int E = 0; E < SW && Pos < VecOut_[I].size(); ++E)
              VecOut_[I][Pos++] = VecOut_[P.SrcBlock][E];
          }
        }
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
      }
    } else if (K == "signal_demux" || K == "signal_switch") {
      // Algebra-only Tier-C stub: passthrough first input. Tier-E adds
      // the proper switch / demux semantics + zero-crossing.
      Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
    } else if (K == "signal_unit_delay" || K == "signal_zoh") {
      // Tier E — read the latched discrete state. The scheduler
      // (`fireDiscreteTicks`) is what updates Z_; the evaluator
      // here just reports the current latched value. Marked as a
      // loop-breaker during lowering, so the topo order never
      // tries to read this block's output before reaching it.
      Out_[I] = Z_[DiscStateOffset_[I]];
    } else if (K == "signal_relay") {
      // Tier E carve-out — hysteretic on/off switch. Z_[Off] holds
      // the latched bit (0.0 = off, 1.0 = on). The evaluator only
      // *reads* Z_; the state update lives in `commitRelayState`,
      // which runs once per major step (after RK4 + fireDiscreteTicks
      // + zero-crossing settle) — otherwise the relay could flip
      // several times per step because stepMajor calls evalAll
      // multiple times for end-of-step refresh.
      double OnV   = paramD(B, "onValue",  1.0);
      double OffV  = paramD(B, "offValue", 0.0);
      Out_[I] = (Z_[DiscStateOffset_[I]] > 0.5) ? OnV : OffV;
    }
    //===-----------------------------------------------------------===//
    // Tier-H — sources, scalar math, lookup tables, routing.
    //===-----------------------------------------------------------===//
    else if (K == "signal_clock") {
      Out_[I] = T;
    } else if (K == "signal_chirp") {
      // Linear frequency sweep f0 → f1 over [0, t1].
      double A  = paramD(B, "amplitude", 1.0);
      double F0 = paramD(B, "f0",        0.1);
      double F1 = paramD(B, "f1",        1.0);
      double T1 = paramD(B, "t1",       10.0);
      double Phase;
      if (T1 > 0.0) {
        double K = (F1 - F0) / T1;
        // Instantaneous phase = 2π·(f0·t + K·t²/2)
        Phase = 2.0 * M_PI * (F0 * T + 0.5 * K * T * T);
      } else {
        Phase = 2.0 * M_PI * F0 * T;
      }
      Out_[I] = A * std::sin(Phase);
    } else if (K == "signal_noise") {
      // xorshift64 + uniform-to-gaussian projection (only when
      // `params.kind == "gaussian"`). Uniform output spans
      // [-amplitude, +amplitude]; gaussian has σ = amplitude.
      double A = paramD(B, "amplitude", 1.0);
      const std::string *Kind = paramS(B, "kind");
      uint64_t &S = NoiseSeed_[I];
      // Advance state — xorshift64.
      S ^= S << 13;
      S ^= S >> 7;
      S ^= S << 17;
      double U1 = (S >> 11) / 9007199254740992.0; // [0, 1)
      if (Kind && *Kind == "gaussian") {
        // Box-Muller: draw a second uniform.
        uint64_t S2 = S ^ 0xDEADBEEFCAFEBABEULL;
        S2 ^= S2 << 13; S2 ^= S2 >> 7; S2 ^= S2 << 17;
        double U2 = (S2 >> 11) / 9007199254740992.0;
        if (U1 < 1e-12) U1 = 1e-12;
        double R = std::sqrt(-2.0 * std::log(U1));
        Out_[I] = A * R * std::cos(2.0 * M_PI * U2);
      } else {
        Out_[I] = A * (2.0 * U1 - 1.0);
      }
    } else if (K == "signal_math_fcn") {
      const std::string *F = paramS(B, "function");
      double U  = inputOf(I, "in");
      double U2 = inputOf(I, "in2");
      double R = U;
      if (F) {
        const std::string &Fn = *F;
        if      (Fn == "sqrt")        R = std::sqrt(U);
        else if (Fn == "exp")         R = std::exp(U);
        else if (Fn == "log")         R = std::log(U);
        else if (Fn == "log10")       R = std::log10(U);
        else if (Fn == "abs")         R = std::fabs(U);
        else if (Fn == "sign")        R = (U > 0) - (U < 0);
        else if (Fn == "square")      R = U * U;
        else if (Fn == "reciprocal")  R = 1.0 / U;
        else if (Fn == "pow")         R = std::pow(U, U2);
        else if (Fn == "hypot")       R = std::hypot(U, U2);
        else if (Fn == "mod")         R = std::fmod(U, U2);
        else if (Fn == "rem")         R = U - U2 * std::trunc(U / U2);
      }
      Out_[I] = R;
    } else if (K == "signal_trig_fcn") {
      const std::string *F = paramS(B, "function");
      double U  = inputOf(I, "in");
      double U2 = inputOf(I, "in2");
      double R = U;
      if (F) {
        const std::string &Fn = *F;
        if      (Fn == "sin")   R = std::sin(U);
        else if (Fn == "cos")   R = std::cos(U);
        else if (Fn == "tan")   R = std::tan(U);
        else if (Fn == "asin")  R = std::asin(U);
        else if (Fn == "acos")  R = std::acos(U);
        else if (Fn == "atan")  R = std::atan(U);
        else if (Fn == "atan2") R = std::atan2(U, U2);
        else if (Fn == "sinh")  R = std::sinh(U);
        else if (Fn == "cosh")  R = std::cosh(U);
        else if (Fn == "tanh")  R = std::tanh(U);
      }
      Out_[I] = R;
    } else if (K == "signal_dead_zone") {
      double U  = inputOf(I, "in");
      double Lo = paramD(B, "lowerLimit", -0.5);
      double Hi = paramD(B, "upperLimit",  0.5);
      if      (U > Hi) Out_[I] = U - Hi;
      else if (U < Lo) Out_[I] = U - Lo;
      else             Out_[I] = 0.0;
    } else if (K == "signal_relop") {
      double U1 = sumInput(I, "in1");
      double U2 = sumInput(I, "in2");
      const std::string *Op = paramS(B, "op");
      bool R = false;
      if (Op) {
        const std::string &O = *Op;
        if      (O == "==" || O == "eq") R = U1 == U2;
        else if (O == "~=" || O == "!=" || O == "ne") R = U1 != U2;
        else if (O == "<"  || O == "lt") R = U1 <  U2;
        else if (O == "<=" || O == "le") R = U1 <= U2;
        else if (O == ">"  || O == "gt") R = U1 >  U2;
        else if (O == ">=" || O == "ge") R = U1 >= U2;
      }
      Out_[I] = R ? 1.0 : 0.0;
    } else if (K == "signal_logical") {
      // Boolean: input ≠ 0 ⇒ true. NOT is unary on `in1`; everything
      // else folds across every connected input port.
      const std::string *Op = paramS(B, "op");
      auto truthy = [](double V) { return V != 0.0; };
      bool R = false;
      if (Op && *Op == "NOT") {
        R = !truthy(sumInput(I, "in1"));
      } else if (Op) {
        std::vector<bool> Vs;
        for (auto &P : Inputs_[I]) Vs.push_back(truthy(Out_[P.SrcBlock]));
        if (Vs.empty()) { R = false; }
        else {
          const std::string &O = *Op;
          if (O == "AND" || O == "and") {
            R = true; for (bool V : Vs) R = R && V;
          } else if (O == "OR" || O == "or") {
            R = false; for (bool V : Vs) R = R || V;
          } else if (O == "NAND" || O == "nand") {
            bool T = true; for (bool V : Vs) T = T && V; R = !T;
          } else if (O == "NOR" || O == "nor") {
            bool T = false; for (bool V : Vs) T = T || V; R = !T;
          } else if (O == "XOR" || O == "xor") {
            int Ones = 0; for (bool V : Vs) Ones += V ? 1 : 0;
            R = (Ones & 1) != 0;
          }
        }
      }
      Out_[I] = R ? 1.0 : 0.0;
    } else if (K == "signal_compare_to_zero" ||
               K == "signal_compare_to_constant") {
      double U  = inputOf(I, "in");
      double C  = (K == "signal_compare_to_constant")
                    ? paramD(B, "constant", 0.0) : 0.0;
      const std::string *Op = paramS(B, "op");
      bool R = false;
      if (Op) {
        const std::string &O = *Op;
        if      (O == "==" || O == "eq") R = U == C;
        else if (O == "~=" || O == "!=" || O == "ne") R = U != C;
        else if (O == "<"  || O == "lt") R = U <  C;
        else if (O == "<=" || O == "le") R = U <= C;
        else if (O == ">"  || O == "gt") R = U >  C;
        else if (O == ">=" || O == "ge") R = U >= C;
      }
      Out_[I] = R ? 1.0 : 0.0;
    } else if (K == "signal_multiport_switch") {
      // First input is the control selector (1-based); the remaining
      // inputs are data lines. Out-of-range selectors fall through to
      // the `defaultOutput` param.
      double Ctrl = inputOf(I, "in1");
      int Idx = static_cast<int>(std::round(Ctrl));
      std::string Port = "in" + std::to_string(Idx + 1);
      bool Found = false;
      double Val = paramD(B, "defaultOutput", 0.0);
      for (auto &P : Inputs_[I])
        if (P.DstPort == Port) { Val = Out_[P.SrcBlock]; Found = true; break; }
      (void)Found;
      Out_[I] = Val;
    } else if (K == "signal_merge") {
      // Output the first non-zero input in port-id order. Matches
      // Simulink's "first-driven-wins" merge semantic for control
      // flows where exactly one source is active at a time.
      double V = paramD(B, "initialOutput", 0.0);
      for (auto &P : Inputs_[I]) {
        double Cand = Out_[P.SrcBlock];
        if (Cand != 0.0) { V = Cand; break; }
      }
      Out_[I] = V;
    } else if (K == "signal_lookup_1d") {
      // Linear interpolation over the cached breakpoint table.
      // Out-of-range inputs clamp to the endpoints (Simulink's
      // "clip" extrapolation; "linear" / "hold" are follow-ups).
      double U = inputOf(I, "in");
      const auto &Tbl = Lookup1DCache_[I];
      if (!Tbl.Valid || Tbl.X.empty()) { Out_[I] = 0.0; }
      else if (U <= Tbl.X.front()) { Out_[I] = Tbl.Y.front(); }
      else if (U >= Tbl.X.back())  { Out_[I] = Tbl.Y.back(); }
      else {
        auto It = std::upper_bound(Tbl.X.begin(), Tbl.X.end(), U);
        size_t K1 = static_cast<size_t>(It - Tbl.X.begin());
        size_t K0 = K1 - 1;
        double Frac = (U - Tbl.X[K0]) / (Tbl.X[K1] - Tbl.X[K0]);
        Out_[I] = Tbl.Y[K0] + Frac * (Tbl.Y[K1] - Tbl.Y[K0]);
      }
    } else if (K == "signal_lookup_2d") {
      // Bilinear interpolation. tableData is row-major, indexed by
      // (i, j) where i runs over breakpointsX and j over breakpointsY.
      double U = inputOf(I, "in1");
      double V = inputOf(I, "in2");
      const auto &Tbl = Lookup2DCache_[I];
      if (!Tbl.Valid) { Out_[I] = 0.0; }
      else {
        auto clamp = [](double X, const std::vector<double> &Bp,
                        size_t &Lo, double &Frac) {
          if (X <= Bp.front()) { Lo = 0; Frac = 0.0; return; }
          if (X >= Bp.back())  { Lo = Bp.size() - 2; Frac = 1.0; return; }
          auto It = std::upper_bound(Bp.begin(), Bp.end(), X);
          Lo = static_cast<size_t>(It - Bp.begin()) - 1;
          Frac = (X - Bp[Lo]) / (Bp[Lo + 1] - Bp[Lo]);
        };
        size_t IX = 0, IY = 0;
        double FX = 0.0, FY = 0.0;
        clamp(U, Tbl.X, IX, FX);
        clamp(V, Tbl.Y, IY, FY);
        size_t W = Tbl.Y.size();
        double Z00 = Tbl.Z[IX       * W + IY];
        double Z10 = Tbl.Z[(IX + 1) * W + IY];
        double Z01 = Tbl.Z[IX       * W + (IY + 1)];
        double Z11 = Tbl.Z[(IX + 1) * W + (IY + 1)];
        double Z0 = Z00 + FX * (Z10 - Z00);
        double Z1 = Z01 + FX * (Z11 - Z01);
        Out_[I] = Z0 + FY * (Z1 - Z0);
      }
    } else if (K == "signal_zero_pole") {
      // Same evaluator as signal_transfer_fcn — the constructor
      // already pre-expanded ZPK into num/den polynomials into the
      // same TFCache_ slot.
      const auto &TF = TFCache_[I];
      int N = static_cast<int>(TF.Den.size()) - 1;
      double U = inputOf(I, "in");
      if (N <= 0 || !TF.Valid) {
        double NumLead = TF.Num.empty() ? 0.0 : TF.Num.front();
        double DenLead = TF.Den.empty() ? 1.0 : TF.Den.front();
        Out_[I] = (NumLead / DenLead) * U;
      } else {
        size_t Off = StateOffset_[I];
        double Lead = TF.Den.front();
        if (Deriv) {
          for (int Ki = 0; Ki < N - 1; ++Ki)
            Deriv[Off + Ki] = State[Off + Ki + 1];
          double Last = U / Lead;
          for (int Ki = 0; Ki < N; ++Ki)
            Last -= (TF.Den[N - Ki] / Lead) * State[Off + Ki];
          Deriv[Off + N - 1] = Last;
        }
        std::vector<double> NPad(N + 1, 0.0);
        int Mdeg = static_cast<int>(TF.Num.size()) - 1;
        for (int Ki = 0; Ki <= Mdeg; ++Ki)
          NPad[N - Mdeg + Ki] = TF.Num[Ki];
        double Y = 0.0;
        for (int Ki = 0; Ki < N; ++Ki)
          // Item-3 — correct controllable-canonical-form output:
          // C = num coefficients (NO division by Lead). The state
          // x is already scaled because B = 1/Lead, so y = C·x
          // gives the right gain. The previous `/Lead` was a bug
          // masked by every demo using `den = "1, ..."` (Lead = 1).
          Y += NPad[N - Ki] * State[Off + Ki];
        Out_[I] = Y;
      }
    } else if (K == "signal_transport_delay") {
      // Linear-interpolate the buffered history at `T - delay`.
      // Buffer is monotonic in time; binary-search the window.
      const auto &Buf = TransportBuf_[I].Samples;
      double TD = TransportBuf_[I].Delay;
      double TTarget = T - TD;
      if (Buf.empty() || TTarget <= Buf.front().first) {
        Out_[I] = TransportBuf_[I].InitialOutput;
      } else if (TTarget >= Buf.back().first) {
        Out_[I] = Buf.back().second;
      } else {
        size_t Lo = 0, Hi = Buf.size() - 1;
        while (Hi - Lo > 1) {
          size_t M = (Lo + Hi) / 2;
          if (Buf[M].first <= TTarget) Lo = M;
          else Hi = M;
        }
        double T0 = Buf[Lo].first,  T1 = Buf[Hi].first;
        double V0 = Buf[Lo].second, V1 = Buf[Hi].second;
        double F = (TTarget - T0) / (T1 - T0);
        Out_[I] = V0 + F * (V1 - V0);
      }
    } else if (K == "signal_discrete_integrator" ||
               K == "signal_discrete_filter" ||
               K == "signal_rate_transition") {
      // All three read the same single-scalar latch as Unit Delay /
      // ZOH. The fireDiscreteTicks scheduler is what advances the
      // state on each sample tick.
      Out_[I] = Z_[DiscStateOffset_[I]];
    } else if (K == "signal_matlab_fcn") {
      // Pack every connected input port (`u1`, `u2`, …) into a
      // flat vector ordered by port id. Two evaluator paths:
      //   - `params.function_body` (Item 4) → walk parsed AST.
      //   - `params.expression`     (Tier-H) → walk expression tree.
      // The function_body path wins when both are present.
      std::vector<std::pair<int, double>> Sorted;
      for (auto &P : Inputs_[I]) {
        if (P.DstPort.size() > 1 && P.DstPort[0] == 'u') {
          try {
            int Idx = std::stoi(P.DstPort.substr(1));
            Sorted.emplace_back(Idx, Out_[P.SrcBlock]);
          } catch (...) {}
        } else if (P.DstPort == "in" || P.DstPort == "in1") {
          Sorted.emplace_back(1, Out_[P.SrcBlock]);
        }
      }
      std::sort(Sorted.begin(), Sorted.end());
      std::vector<double> U;
      U.reserve(Sorted.size());
      for (auto &PR : Sorted) U.push_back(PR.second);
      if (MatlabFnCache_[I]) {
        Out_[I] = runMatlabFunction(*MatlabFnCache_[I], U, T);
      } else {
        Out_[I] = evalMatlabFcn(MatlabFcnCache_[I].get(), U, T);
      }
    } else {
      // Loader-level reserved kinds are rejected at lowering, so
      // anything reaching here is an evaluator gap — treat as
      // passthrough so simulation doesn't crash and the user sees
      // the wrong-but-finite result instead of a segfault.
      Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
    }
  }
}

void MflowLinkSim::derivative(double T, const double *State, double *Deriv) {
  evalAll(T, State, Deriv);
  // Item-2 — algebraic-loop solver. After the topological pass has
  // written every output, re-iterate the loop members until their
  // outputs converge within `relTol`. Each iteration calls the
  // per-block evaluator again with the current `Out_` snapshot;
  // the loop's wiring (Inputs_) reaches back into the same Out_,
  // so the next iteration sees the previous one's outputs. This is
  // the classic Picard / fixed-point method; under contraction it
  // converges (Banach), and a true Newton / trust-region solver is
  // a deeper follow-up. Diverging loops are detected after `MaxIt`
  // and reported via `consumeAlgebraicLoopFailures` (collected for
  // the DAP server to surface as `stopped { reason: "algebraic
  // loop did not converge" }`).
  if (M_.AlgebraicLoops.empty()) return;
  const double Tol = std::max(M_.Solver.RelTol, 1e-8);
  const int MaxIt = 50;
  for (auto &Loop : M_.AlgebraicLoops) {
    if (Loop.Members.empty()) continue;
    std::vector<double> Prev(Loop.Members.size());
    bool Converged = false;
    for (int It = 0; It < MaxIt; ++It) {
      for (size_t K = 0; K < Loop.Members.size(); ++K)
        Prev[K] = Out_[Loop.Members[K]];
      // Re-evaluate just the loop members in their stored order.
      // We piggy-back on the existing evalAll by re-running the
      // full pass — cheap for our typical block counts and avoids
      // duplicating the per-kind dispatch. Refactoring to a
      // per-block evaluator entry point is a follow-up.
      evalAll(T, State, nullptr);
      double Delta = 0.0;
      for (size_t K = 0; K < Loop.Members.size(); ++K) {
        double D = Out_[Loop.Members[K]] - Prev[K];
        if (D < 0) D = -D;
        if (D > Delta) Delta = D;
      }
      if (Delta < Tol) { Converged = true; break; }
    }
    if (!Converged) {
      AlgLoopFailures_.push_back({T, Loop.Members});
    }
  }
  // Refresh derivatives one final time so the RK4 stages see the
  // settled outputs — without this, an integrator immediately
  // downstream of a loop would integrate the pre-converged input.
  if (Deriv) evalAll(T, State, Deriv);
}

//===----------------------------------------------------------------------===//
// Classic RK4 fixed-step integrator.
//
// Picked over Dormand-Prince for Tier C because it's the smallest
// thing that gives the demos sensible answers; the existing matlab_*
// ode45 builtins can be plumbed in once `-emit-mflowlink-cpp` (Tier G)
// lands. The user-facing `solver.algorithm` field is passed through
// the IR untouched so the choice surfaces in `--dry-run` output.
//===----------------------------------------------------------------------===//

// Take a single RK4 substep of size `H` from time `TBegin` with the
// state in `YIn`. Writes the integrated state to `YOut`. No logging,
// no scheduler — just the pure integrator.
static void rk4Substep(MflowLinkSim &Sim,
                       void (MflowLinkSim::*Deriv)(double, const double *,
                                                   double *),
                       double TBegin, double H,
                       const std::vector<double> &YIn,
                       std::vector<double> &YOut,
                       std::vector<double> &K1,
                       std::vector<double> &K2,
                       std::vector<double> &K3,
                       std::vector<double> &K4,
                       std::vector<double> &Yt) {
  const size_t Nx = YIn.size();
  (Sim.*Deriv)(TBegin, YIn.data(), K1.data());
  for (size_t I = 0; I < Nx; ++I) Yt[I] = YIn[I] + 0.5 * H * K1[I];
  (Sim.*Deriv)(TBegin + 0.5 * H, Yt.data(), K2.data());
  for (size_t I = 0; I < Nx; ++I) Yt[I] = YIn[I] + 0.5 * H * K2[I];
  (Sim.*Deriv)(TBegin + 0.5 * H, Yt.data(), K3.data());
  for (size_t I = 0; I < Nx; ++I) Yt[I] = YIn[I] + H * K3[I];
  (Sim.*Deriv)(TBegin + H, Yt.data(), K4.data());
  for (size_t I = 0; I < Nx; ++I)
    YOut[I] = YIn[I] + (H / 6.0) * (K1[I] + 2.0 * K2[I]
                                      + 2.0 * K3[I] + K4[I]);
}

//===-----------------------------------------------------------------===//
// Item-2 — Dormand-Prince 5(4) adaptive integrator (the standard
// `ode45` tableau). One call attempts a single step of size `H`,
// returns both the 5th-order solution in `YOut` and the embedded
// 4th-order error estimate in `Err`. The caller compares `Err`
// against the configured `relTol` / `absTol` and either accepts +
// grows the step, or rejects + shrinks. See e.g. Hairer & Wanner
// "Solving ODEs I" §II.4. Coefficients are the Dormand-Prince
// (1980) tableau used by MATLAB's `ode45` and SciPy's `RK45`.
//===-----------------------------------------------------------------===//

namespace {
struct DOPRI5Workspace {
  std::vector<double> K1, K2, K3, K4, K5, K6, K7, Yt;
};

void dopri5Step(MflowLinkSim &Sim,
                void (MflowLinkSim::*Deriv)(double, const double *, double *),
                double TBegin, double H,
                const std::vector<double> &YIn,
                std::vector<double> &YOut,
                std::vector<double> &Err,
                DOPRI5Workspace &W) {
  const size_t Nx = YIn.size();
  if (W.K1.size() != Nx) {
    W.K1.assign(Nx, 0.0); W.K2.assign(Nx, 0.0); W.K3.assign(Nx, 0.0);
    W.K4.assign(Nx, 0.0); W.K5.assign(Nx, 0.0); W.K6.assign(Nx, 0.0);
    W.K7.assign(Nx, 0.0); W.Yt.assign(Nx, 0.0);
  }
  // Stage 1: k1 = f(t, y).
  (Sim.*Deriv)(TBegin, YIn.data(), W.K1.data());
  // Stage 2: t + h/5, y + h·(1/5)·k1.
  for (size_t I = 0; I < Nx; ++I) W.Yt[I] = YIn[I] + H * (1.0 / 5.0) * W.K1[I];
  (Sim.*Deriv)(TBegin + 0.2 * H, W.Yt.data(), W.K2.data());
  // Stage 3: t + 3h/10, y + h·(3/40 k1 + 9/40 k2).
  for (size_t I = 0; I < Nx; ++I)
    W.Yt[I] = YIn[I] + H * ((3.0 / 40.0) * W.K1[I] + (9.0 / 40.0) * W.K2[I]);
  (Sim.*Deriv)(TBegin + 0.3 * H, W.Yt.data(), W.K3.data());
  // Stage 4: t + 4h/5.
  for (size_t I = 0; I < Nx; ++I)
    W.Yt[I] = YIn[I] + H * ((44.0 / 45.0) * W.K1[I]
                            - (56.0 / 15.0) * W.K2[I]
                            + (32.0 / 9.0)  * W.K3[I]);
  (Sim.*Deriv)(TBegin + 0.8 * H, W.Yt.data(), W.K4.data());
  // Stage 5: t + 8h/9.
  for (size_t I = 0; I < Nx; ++I)
    W.Yt[I] = YIn[I] + H * ((19372.0 / 6561.0)  * W.K1[I]
                            - (25360.0 / 2187.0) * W.K2[I]
                            + (64448.0 / 6561.0) * W.K3[I]
                            - (212.0   / 729.0)  * W.K4[I]);
  (Sim.*Deriv)(TBegin + (8.0 / 9.0) * H, W.Yt.data(), W.K5.data());
  // Stage 6: t + h. Combines into the 5th-order solution.
  for (size_t I = 0; I < Nx; ++I)
    W.Yt[I] = YIn[I] + H * ((9017.0 / 3168.0)    * W.K1[I]
                            - (355.0 / 33.0)     * W.K2[I]
                            + (46732.0 / 5247.0) * W.K3[I]
                            + (49.0 / 176.0)     * W.K4[I]
                            - (5103.0 / 18656.0) * W.K5[I]);
  (Sim.*Deriv)(TBegin + H, W.Yt.data(), W.K6.data());
  // 5th-order solution: y_{n+1}.
  for (size_t I = 0; I < Nx; ++I)
    YOut[I] = YIn[I] + H * ((35.0 / 384.0)      * W.K1[I]
                            + (500.0 / 1113.0)  * W.K3[I]
                            + (125.0 / 192.0)   * W.K4[I]
                            - (2187.0 / 6784.0) * W.K5[I]
                            + (11.0 / 84.0)     * W.K6[I]);
  // Stage 7: FSAL — k7 evaluated at YOut to feed the embedded
  // 4th-order solution.
  (Sim.*Deriv)(TBegin + H, YOut.data(), W.K7.data());
  // Embedded error estimate: yhat_{n+1} = y_n + h·Σ b̂ᵢ·kᵢ; the
  // difference `YOut - yhat` is what we report.
  Err.assign(Nx, 0.0);
  for (size_t I = 0; I < Nx; ++I) {
    double E = H * ((35.0 / 384.0     - 5179.0 / 57600.0)    * W.K1[I]
                  + (500.0 / 1113.0   - 7571.0 / 16695.0)    * W.K3[I]
                  + (125.0 / 192.0    - 393.0 / 640.0)       * W.K4[I]
                  + (-2187.0 / 6784.0 - (-92097.0 / 339200.0)) * W.K5[I]
                  + (11.0 / 84.0      - 187.0 / 2100.0)      * W.K6[I]
                  + (0.0              - 1.0 / 40.0)          * W.K7[I]);
    Err[I] = E;
  }
}
} // namespace

double MflowLinkSim::stepMajor() {
  // Pick `h` as the smaller of the configured fixed step and the
  // distance to the next discrete event / stopTime — the multirate
  // scheduler's job (§7.1). The simulation never integrates past a
  // discrete tick, so each discrete partition fires at exact period
  // boundaries and the continuous partitions only see their own
  // smooth segment of the trajectory.
  const size_t Nx = Y_.size();
  double H = StepSize_;
  double TNextDisc = std::numeric_limits<double>::infinity();
  for (size_t I = 0; I < NextFire_.size(); ++I)
    if (NextFire_[I] < TNextDisc) TNextDisc = NextFire_[I];
  double TTarget = T_ + H;
  if (TNextDisc < TTarget) TTarget = TNextDisc;
  if (TTarget > M_.Solver.StopTime) TTarget = M_.Solver.StopTime;
  H = TTarget - T_;
  if (H <= 1e-12) {
    // No continuous integration to do, but a discrete tick might
    // still be exactly due at T_ — fire it and call this a step.
    if (TNextDisc <= T_ + 1e-12) {
      pushSnapshot();
      fireDiscreteTicks();
      evalAll(T_, Y_.data(), nullptr);
      ++MajorSteps_;
      logSample();
      return 0.0;
    }
    return 0.0;
  }

  // Snapshot BEFORE we step so step-back can restore exactly here.
  // After a step-back the user is rewriting an alternate future:
  // the log columns were already truncated by `stepBackMajor`.
  LogsTruncated_ = false;
  pushSnapshot();

  // Remember start-of-step state for zero-crossing bisection.
  std::vector<double> Y0 = Y_;

  std::vector<double> K1(Nx), K2(Nx), K3(Nx), K4(Nx), Yt(Nx), Y1(Nx);
  if (Nx > 0) {
    if (AdaptiveSolver_) {
      // Item-2 — Dormand-Prince RK4(5) with PI step-size control.
      // Try the full window `H`; if the embedded error estimate
      // exceeds tolerance, shrink and retry. Each accepted step
      // updates `CurrentAdaptiveH_` so the next major step starts
      // from a reasonable size. The maxStep cap (`StepSize_`) is
      // respected: the chosen step never exceeds it.
      static thread_local DOPRI5Workspace WS;
      std::vector<double> Err(Nx, 0.0);
      double TLocal = T_;
      double TEnd   = T_ + H;
      double HCur   = std::min(CurrentAdaptiveH_, H);
      const double RelTol = M_.Solver.RelTol;
      const double AbsTol = M_.Solver.AbsTol;
      const int MaxRetries = 32;
      while (TLocal < TEnd - 1e-15) {
        double HTry = std::min(HCur, TEnd - TLocal);
        if (HTry <= 1e-15) break;
        int Retry = 0;
        bool Accepted = false;
        while (!Accepted && Retry++ < MaxRetries) {
          dopri5Step(*this, &MflowLinkSim::derivative,
                     TLocal, HTry, Y_, Y1, Err, WS);
          // Norm: max over elements of |err_i| / (atol + rtol·|y_i|).
          double Norm = 0.0;
          for (size_t I = 0; I < Nx; ++I) {
            double Sc = AbsTol + RelTol *
                        std::max(std::fabs(Y_[I]), std::fabs(Y1[I]));
            if (Sc <= 0.0) Sc = AbsTol > 0 ? AbsTol : 1e-12;
            double E = std::fabs(Err[I]) / Sc;
            if (E > Norm) Norm = E;
          }
          if (Norm <= 1.0) {
            Accepted = true;
            // PI-control: grow step on success, capped at the
            // configured maxStep.
            double Factor = (Norm > 1e-12)
                              ? 0.9 * std::pow(Norm, -0.2)
                              : 5.0;
            if (Factor > 5.0) Factor = 5.0;
            HCur = std::min(HTry * Factor, StepSize_);
          } else {
            // Reject + shrink. Lower bound keeps us from
            // infinite-looping on a hard discontinuity.
            double Factor = 0.9 * std::pow(Norm, -0.2);
            if (Factor < 0.1) Factor = 0.1;
            HTry = HTry * Factor;
            if (HTry < 1e-15) break;
          }
        }
        if (!Accepted) {
          // Last-resort fallback — take a fixed RK4 step at HTry
          // and move on. Better an inaccurate step than a hang.
          rk4Substep(*this, &MflowLinkSim::derivative, TLocal, HTry,
                     Y_, Y1, K1, K2, K3, K4, Yt);
        }
        Y_ = Y1;
        TLocal += HTry;
      }
      CurrentAdaptiveH_ = HCur;
    } else {
      rk4Substep(*this, &MflowLinkSim::derivative, T_, H, Y_, Y1,
                 K1, K2, K3, K4, Yt);
      Y_ = std::move(Y1);
    }
  }
  T_ += H;
  ++MajorSteps_;

  // Refresh outputs at the new time so zero-crossing predicates and
  // discrete-input reads see a coherent post-step picture.
  evalAll(T_, Y_.data(), nullptr);

  // Zero-crossing detection + bisection. We only bisect the *first*
  // observed sign flip per major step; multiple coincident crossings
  // surface as separate events on subsequent steps once the bisected
  // state has been resumed.
  for (size_t K = 0; K < M_.ZeroCrossings.size(); ++K) {
    int Now = predicateSign(K);
    int Was = ZCSign_[K];
    // Any sign change is a crossing — including the band-entry /
    // band-exit transitions (0 → ±1, ±1 → 0). Saturation sits at
    // sign 0 while inside its rails, and the moment we care about is
    // when it engages or releases a rail.
    if (Was != Now && Nx > 0) {
      double TStar = bisectZeroCrossing(K, T_ - H, Y0, T_);
      ZCQueue_.push_back({M_.ZeroCrossings[K].BlockId, TStar});
      // The bisector left Y_ at the crossing time and T_ at TStar —
      // refresh outputs once more so logging records the crossing
      // instant, not the post-overshoot value.
      T_ = TStar;
      evalAll(T_, Y_.data(), nullptr);
      ZCSign_[K] = predicateSign(K);
      break;
    }
    ZCSign_[K] = Now;
  }

  // Fire any discrete blocks whose period boundary we just landed on.
  fireDiscreteTicks();
  // Outputs again, so a Unit Delay / ZOH that latched a new value at
  // this tick is visible to downstream blocks in the logged sample.
  evalAll(T_, Y_.data(), nullptr);
  // Tier E carve-out — once-per-step relay sweep. Runs after every
  // other update so its input read sees the post-discrete-tick
  // outputs, but before logging so the latched output appears in
  // the recorded sample at the same `t` it actually flipped on.
  commitRelayState();
  evalAll(T_, Y_.data(), nullptr);
  // Tier F carve-out — snapshot the just-finished outputs as the
  // *previous* values for the next step's edge-trigger detection.
  // Doing this after the final evalAll means PrevOut_ holds the
  // logged value, so a rising edge between consecutive logged
  // samples is what the next step's evalAll keys on.
  PrevOut_ = Out_;
  // Tier-H — append every transport-delay block's current input to
  // its history buffer. We do this at end-of-step so the recorded
  // sample's `t` is the just-advanced T_, matching how Out_ /
  // Y_ are sampled. Trim ancient samples older than `delay + 2·h`
  // so the buffer doesn't grow unboundedly on long runs.
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    if (M_.Blocks[I].Kind != "signal_transport_delay") continue;
    double Uin = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in") Uin = Out_[P.SrcBlock];
    auto &Buf = TransportBuf_[I].Samples;
    Buf.push_back({T_, Uin});
    double Keep = T_ - TransportBuf_[I].Delay - 2.0 * StepSize_;
    size_t Drop = 0;
    while (Drop + 1 < Buf.size() && Buf[Drop + 1].first < Keep) ++Drop;
    if (Drop > 0) Buf.erase(Buf.begin(), Buf.begin() + Drop);
  }
  logSample();
  return H;
}

void MflowLinkSim::commitRelayState() {
  // Idempotent per call: only flips when the input has crossed
  // *outside* the dead-band — repeated calls with the same input
  // converge to the same latched state on the first pass.
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const auto &B = M_.Blocks[I];
    if (B.Kind != "signal_relay") continue;
    double U = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in") U += Out_[P.SrcBlock];
    double OnPt  = paramD(B, "onPoint",  0.5);
    double OffPt = paramD(B, "offPoint", -0.5);
    size_t Off = DiscStateOffset_[I];
    if (Z_[Off] <= 0.5 && U > OnPt)       Z_[Off] = 1.0;
    else if (Z_[Off] >  0.5 && U < OffPt) Z_[Off] = 0.0;
  }
}

void MflowLinkSim::fireDiscreteTicks() {
  const double Eps = 1e-12;
  bool AnyFired = false;
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const auto &B = M_.Blocks[I];
    if (B.SampleClass != SampleTimeClass::Discrete) continue;
    if (NextFire_[I] > T_ + Eps) continue;
    // Read the block's current input — same wiring as evalAll.
    double U = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in") U = Out_[P.SrcBlock];
    size_t Off = DiscStateOffset_[I];
    if (B.Kind == "signal_zoh" || B.Kind == "signal_rate_transition") {
      // Held value updates at this tick and remains until the next.
      Z_[Off] = U;
      Znext_[Off] = U;
    } else if (B.Kind == "signal_unit_delay") {
      // One-tick lag: stage the new value; commit at end-of-tick.
      Znext_[Off] = U;
    } else if (B.Kind == "signal_discrete_integrator") {
      // §17.5 #4 — proper Forward / Backward Euler / Trapezoidal.
      // `U` here is the input value AT tick time t = (n+1)·h (the
      // fire is at end-of-step). DiscPrevU_[I] retains u[n] from
      // the previous tick.
      //   Forward Euler:  y[n+1] = y[n] + h · u[n]      (uses PrevU)
      //   Backward Euler: y[n+1] = y[n] + h · u[n+1]    (uses U)
      //   Trapezoidal:    y[n+1] = y[n] + h/2·(u[n]+u[n+1])
      double H = B.SamplePeriod > 0.0 ? B.SamplePeriod : 1.0;
      const std::string *MS = nullptr;
      auto It = B.Params.find("method");
      if (It != B.Params.end()) MS = &It->second;
      double UPrev = DiscPrevU_[I];
      double Y = Z_[Off];
      if (MS && (*MS == "BackwardEuler" || *MS == "backward_euler")) {
        Y = Y + H * U;
      } else if (MS && (*MS == "Trapezoidal" || *MS == "trapezoidal")) {
        Y = Y + 0.5 * H * (UPrev + U);
      } else {
        // Default = Forward Euler (the unflagged case).
        Y = Y + H * UPrev;
      }
      Z_[Off]     = Y;
      Znext_[Off] = Y;
      DiscPrevU_[I] = U;
    } else if (B.Kind == "signal_discrete_filter") {
      // Direct-form-II IIR step: y[n] = (num · u_history − den[1..] · y_history) / den[0]
      // For our single-output Tier-H pass, Z_ stores `den.size()-1`
      // taps of y. We re-derive the state every tick from the
      // already-cached coefficients.
      const auto &TF = TFCache_[I];
      int N = static_cast<int>(TF.Den.size()) - 1;
      double Y = 0.0;
      if (TF.Valid && N >= 0) {
        double Lead = TF.Den.front();
        double NumLead = TF.Num.empty() ? 0.0 : TF.Num.front();
        // Use the leading-coefficient feedforward of u + the running
        // sum of previous-y feedback taps. A full direct-form-II
        // needs both u-history and y-history; the IIR Tier-H pass
        // implements the y-feedback half (poles) so simple low/
        // high-pass IIR filters work. Zero-only FIR designs are a
        // follow-up that needs a u-history buffer too.
        Y = (NumLead / Lead) * U;
        for (int K = 0; K < N; ++K)
          Y -= (TF.Den[K + 1] / Lead) * Z_[Off + K];
      } else {
        Y = U;
      }
      // Shift taps: y[n-1] ← y[n], y[n-2] ← y[n-1], …
      for (int K = N - 1; K >= 1; --K) Z_[Off + K] = Z_[Off + K - 1];
      if (N >= 1) Z_[Off + 0] = Y;
      // Even if N == 0 (static gain), Out_ still needs to surface Y
      // — store it in slot 0 for the evaluator to read.
      if (N == 0) Z_[Off] = Y;
    }
    NextFire_[I] += B.SamplePeriod > 0.0 ? B.SamplePeriod : 1.0;
    AnyFired = true;
  }
  if (AnyFired) {
    // Commit the unit-delay shadow buffer. We do this once after
    // every discrete tick at this time, so a chain of unit delays
    // all advance by exactly one tick per period boundary.
    for (size_t I = 0; I < M_.Blocks.size(); ++I) {
      const auto &B = M_.Blocks[I];
      if (B.Kind != "signal_unit_delay") continue;
      size_t Off = DiscStateOffset_[I];
      Z_[Off] = Znext_[Off];
    }
  }
}

int MflowLinkSim::predicateSign(size_t K) const {
  if (K >= M_.ZeroCrossings.size()) return 0;
  const auto &ZC = M_.ZeroCrossings[K];
  const MflBlock *B = M_.findBlock(ZC.BlockId);
  if (!B) return 0;
  size_t I = 0;
  for (; I < M_.Blocks.size(); ++I)
    if (&M_.Blocks[I] == B) break;
  if (I == M_.Blocks.size()) return 0;
  if (B->Kind == "signal_saturation") {
    // Two rails worth caring about, combined into one predicate that
    // flips when the *clamping* engages or releases. This collapses
    // to a clean sign change exactly when the input crosses either
    // limit — enough granularity for the demo + scope highlight.
    double U = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in") U += Out_[P.SrcBlock];
    double Up = paramD(*B, "upperLimit",  1.0);
    double Lo = paramD(*B, "lowerLimit", -1.0);
    if (U > Up) return +1;
    if (U < Lo) return -1;
    return 0;
  }
  if (B->Kind == "signal_switch") {
    double Ctrl = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in2") Ctrl += Out_[P.SrcBlock];
    double Th = paramD(*B, "threshold", 0.0);
    double D = Ctrl - Th;
    return D > 0 ? +1 : (D < 0 ? -1 : 0);
  }
  if (B->Kind == "signal_relay") {
    // Predicate flips sign at each rail crossing. Inside the dead-
    // band the relay's state is sticky so the predicate sign just
    // tracks the input's *current* zone (above onPoint, below
    // offPoint, or in-between). Either flip → a registered ZC the
    // bisector will land on.
    double U = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in") U += Out_[P.SrcBlock];
    double OnPt  = paramD(*B, "onPoint",  0.5);
    double OffPt = paramD(*B, "offPoint", -0.5);
    if (U > OnPt)  return +1;
    if (U < OffPt) return -1;
    return 0;
  }
  return 0;
}

double MflowLinkSim::bisectZeroCrossing(size_t K, double TStart,
                                        const std::vector<double> &YStart,
                                        double TEnd) {
  // Illinois-flavoured bisection. We have a sign flip on `[TStart,
  // TEnd]`; integrate forward from `YStart` to a midpoint, check the
  // sign, and tighten the bracket. The simulator already trusts the
  // RK4 integrator inside this step so re-integration is exact at
  // the step size we picked.
  std::vector<double> K1(YStart.size()), K2(YStart.size()),
      K3(YStart.size()), K4(YStart.size()), Yt(YStart.size()),
      YMid(YStart.size());
  double Lo = TStart, Hi = TEnd;
  std::vector<double> YHi = Y_; // post-step state already in Y_
  int SignLo = ZCSign_[K];
  for (int Iter = 0; Iter < 32; ++Iter) {
    double Mid = 0.5 * (Lo + Hi);
    if (Hi - Lo < 1e-9) {
      Y_ = YHi;
      return Mid;
    }
    rk4Substep(*this, &MflowLinkSim::derivative, Lo, Mid - Lo, YStart,
               YMid, K1, K2, K3, K4, Yt);
    // Pin Y_ to the midpoint state so predicateSign + evalAll read
    // the correct inputs at `Mid`.
    Y_ = YMid;
    evalAll(Mid, Y_.data(), nullptr);
    int SignMid = predicateSign(K);
    if (SignMid == 0) return Mid;
    if (SignMid == SignLo) {
      Lo = Mid;
      // YStart stays the start point of the next sub-integration —
      // but to integrate from a closer-to-the-root state, capture
      // `YMid` as the new start; the function-arg is `const &` so
      // we'd need a mutable copy. Cheaper: keep integrating from
      // YStart over the full `Mid - Lo` interval. Numerically still
      // converges quadratically because each iteration halves the
      // bracket; integration over a smaller `h` is more accurate,
      // not less.
    } else {
      Hi = Mid;
      YHi = YMid;
    }
  }
  Y_ = YHi;
  return Hi;
}

std::vector<MflowLinkSim::CrossingEvent>
MflowLinkSim::consumeZeroCrossings() {
  std::vector<CrossingEvent> Out;
  Out.swap(ZCQueue_);
  return Out;
}

std::vector<MflowLinkSim::AlgebraicLoopFailure>
MflowLinkSim::consumeAlgebraicLoopFailures() {
  std::vector<AlgebraicLoopFailure> Out;
  Out.swap(AlgLoopFailures_);
  return Out;
}

std::string MflowLinkSim::stepBlock() {
  // The block-stepping cursor is purely informational — it does not
  // hold state-update aside from the index, because the simulator
  // computes block outputs as a single topo-ordered pass. The cursor
  // controls *which* block the IDE highlights; advancing past the
  // last block commits the major step and resets the cursor.
  const size_t N = M_.ExecOrder.size();
  if (BlockCursor_ < N) {
    size_t I = M_.ExecOrder[BlockCursor_];
    ++BlockCursor_;
    return M_.Blocks[I].Id;
  }
  // Cursor wrapped — commit one major step.
  stepMajor();
  BlockCursor_ = 0;
  return std::string{};
}

std::string MflowLinkSim::stepBackBlock() {
  if (BlockCursor_ > 0) {
    --BlockCursor_;
    size_t I = M_.ExecOrder[BlockCursor_];
    return M_.Blocks[I].Id;
  }
  // Cursor at zero — pop a major step instead.
  if (!stepBackMajor()) return std::string{};
  BlockCursor_ = M_.ExecOrder.size();
  return std::string{};
}

std::string MflowLinkSim::activeBlockId() const {
  const size_t N = M_.ExecOrder.size();
  if (BlockCursor_ == 0 || BlockCursor_ > N) return std::string{};
  size_t I = M_.ExecOrder[BlockCursor_ - 1];
  return M_.Blocks[I].Id;
}

void MflowLinkSim::runToCompletion() {
  reset();
  // A returned `h = 0` is ambiguous in a multi-rate world: it can
  // mean "stuck" (StopTime reached, no work left), or "a discrete
  // tick fired exactly at the current time and the next continuous
  // segment hasn't started yet". Disambiguate by tracking
  // MajorSteps_ — a tick fire still bumps it, so a `(h, ΔSteps) =
  // (0, 0)` pair is the only true stall.
  while (T_ < M_.Solver.StopTime - 1e-15) {
    size_t Before = MajorSteps_;
    double H = stepMajor();
    if (H <= 0.0 && MajorSteps_ == Before) break;
  }
}

//===----------------------------------------------------------------------===//
// Log + CSV
//===----------------------------------------------------------------------===//

void MflowLinkSim::logSample() {
  for (size_t Ci = 0; Ci < LogBlocks_.size(); ++Ci) {
    size_t I = LogBlocks_[Ci];
    int E = LogElements_[Ci];
    double V;
    if (OutWidth_[I] > 1 && E < (int)VecOut_[I].size()) {
      V = VecOut_[I][E];
    } else {
      V = Out_[I];
    }
    LogColumns_[Ci].push_back({T_, V});
  }
}

void MflowLinkSim::writeCsv(std::ostream &OS) const {
  OS << "t";
  for (auto &N : LogNames_) OS << "," << N;
  OS << "\n";
  size_t Rows = LogColumns_.empty() ? 0 : LogColumns_.front().size();
  OS.setf(std::ios::scientific);
  OS.precision(9);
  for (size_t R = 0; R < Rows; ++R) {
    OS << LogColumns_.front()[R].T;
    for (auto &C : LogColumns_) OS << "," << C[R].Value;
    OS << "\n";
  }
}

void MflowLinkSim::pushSnapshot() {
  if (SnapshotCap_ == 0) return;
  Snapshot S;
  S.T = T_;
  S.MajorSteps = MajorSteps_;
  S.Y = Y_;
  S.Out = Out_;
  S.Z = Z_;
  S.DiscPrevU = DiscPrevU_;
  S.PrevOut = PrevOut_;
  S.NextFire = NextFire_;
  S.ZCSign = ZCSign_;
  S.LogRows = LogColumns_.empty() ? 0 : LogColumns_.front().size();
  if (Snapshots_.size() >= SnapshotCap_)
    Snapshots_.erase(Snapshots_.begin());
  Snapshots_.push_back(std::move(S));
}

bool MflowLinkSim::stepBackMajor() {
  if (Snapshots_.empty()) return false;
  Snapshot S = std::move(Snapshots_.back());
  Snapshots_.pop_back();
  T_ = S.T;
  MajorSteps_ = S.MajorSteps;
  Y_ = std::move(S.Y);
  Out_ = std::move(S.Out);
  Z_ = std::move(S.Z);
  Znext_ = Z_; // shadow buffer matches the latched value on restore
  DiscPrevU_ = std::move(S.DiscPrevU);
  PrevOut_ = std::move(S.PrevOut);
  NextFire_ = std::move(S.NextFire);
  ZCSign_ = std::move(S.ZCSign);
  // Truncate any log rows newer than this snapshot — they belong to
  // the now-rewound timeline.
  for (auto &C : LogColumns_)
    if (C.size() > S.LogRows) C.resize(S.LogRows);
  LogsTruncated_ = true;
  return true;
}

std::vector<std::pair<std::string, double>>
MflowLinkSim::currentLoggedOutputs() const {
  std::vector<std::pair<std::string, double>> Out;
  Out.reserve(LogBlocks_.size());
  for (size_t Ci = 0; Ci < LogBlocks_.size(); ++Ci) {
    size_t I = LogBlocks_[Ci];
    int E = LogElements_[Ci];
    double V = (OutWidth_[I] > 1 && E < (int)VecOut_[I].size())
                 ? VecOut_[I][E] : Out_[I];
    Out.emplace_back(LogNames_[Ci], V);
  }
  return Out;
}

std::string validateMatlabFcnExpression(const std::string &Expr) {
  ExprParser P(Expr);
  std::string Err;
  auto Tree = P.parse(Err);
  if (Tree) return std::string{};
  return Err.empty() ? std::string("invalid expression") : Err;
}

//===----------------------------------------------------------------------===//
// Item-4 — Real MATLAB Function block.
//
// `params.function_body` carries a full `function y = f(u1, u2, ...);
// ... end` definition that we parse via the existing matlab_llvm
// lexer + parser, then walk with a small scalar AST interpreter at
// runtime. Compared to the (already-shipped) `params.expression`
// path, this supports multi-statement bodies with assignments and
// `if`/`else` control flow.
//
// Supported AST node subset (anything else → sourced diagnostic
// at lowering time):
//   - IntegerLiteral, FPLiteral, NameExpr
//   - BinaryOpExpr: + - * / ^ .* ./ .^ == ~= < <= > >= && || & |
//   - UnaryOpExpr: + - ~
//   - CallOrIndex (treated as a builtin math call — no user calls)
//   - AssignStmt (single LHS NameExpr only — no multi-return,
//     no indexing)
//   - ExprStmt
//   - IfStmt with elseifs / else
//   - Block, ReturnStmt
//
// Vectors, matrices, cells, strings, structs, while, for, switch,
// try, recursion, multi-return — all out of scope for this MVP.
//===----------------------------------------------------------------------===//

struct MflowLinkSim::MatlabFunctionState {
  std::unique_ptr<matlab::SourceManager> SM;
  std::unique_ptr<matlab::DiagnosticEngine> Diag;
  std::unique_ptr<matlab::ASTContext> AST;
  matlab::Function *Fn = nullptr;
  std::vector<std::string> InNames;
  std::string OutName;
};

MflowLinkSim::~MflowLinkSim() = default;

namespace {

// Parse a `params.function_body` blob into a fresh state. Returns
// `(state, error_message)`. On any error the state is left empty
// and the message is non-empty.
std::pair<std::unique_ptr<MflowLinkSim::MatlabFunctionState>, std::string>
parseMatlabFunctionBody(const std::string &Source) {
  using namespace matlab;
  auto S = std::make_unique<MflowLinkSim::MatlabFunctionState>();
  S->SM = std::make_unique<SourceManager>();
  S->Diag = std::make_unique<DiagnosticEngine>(*S->SM);
  S->AST = std::make_unique<ASTContext>();
  FileID F = S->SM->addBuffer("<signal_matlab_fcn>", Source);
  Lexer L(*S->SM, F, *S->Diag);
  auto Tokens = L.tokenize();
  if (S->Diag->hasErrors())
    return {nullptr, "lexer error in function_body"};
  Parser P(std::move(Tokens), *S->AST, *S->Diag);
  TranslationUnit *TU = P.parseFile();
  if (!TU || S->Diag->hasErrors())
    return {nullptr, "parse error in function_body"};
  if (TU->Functions.size() != 1)
    return {nullptr, "function_body must contain exactly one function"};
  S->Fn = TU->Functions.front();
  if (S->Fn->Outputs.size() != 1)
    return {nullptr,
            "function_body must declare exactly one output variable"};
  S->OutName = std::string(S->Fn->Outputs[0]);
  S->InNames.reserve(S->Fn->Inputs.size());
  for (auto &In : S->Fn->Inputs) S->InNames.emplace_back(In);
  return {std::move(S), std::string{}};
}

//===-----------------------------------------------------------------===//
// Scalar AST interpreter for `signal_matlab_fcn` / function_body.
//===-----------------------------------------------------------------===//

struct InterpEnv {
  std::map<std::string, double> Vars;
};
struct ReturnSignal {};

double interpExpr(const matlab::Expr *E, InterpEnv &Env);

double callBuiltinScalar(const std::string &Name,
                         const std::vector<double> &Args) {
  auto arg = [&](size_t I) { return I < Args.size() ? Args[I] : 0.0; };
  if (Name == "sin")    return std::sin(arg(0));
  if (Name == "cos")    return std::cos(arg(0));
  if (Name == "tan")    return std::tan(arg(0));
  if (Name == "asin")   return std::asin(arg(0));
  if (Name == "acos")   return std::acos(arg(0));
  if (Name == "atan")   return std::atan(arg(0));
  if (Name == "atan2")  return std::atan2(arg(0), arg(1));
  if (Name == "sinh")   return std::sinh(arg(0));
  if (Name == "cosh")   return std::cosh(arg(0));
  if (Name == "tanh")   return std::tanh(arg(0));
  if (Name == "exp")    return std::exp(arg(0));
  if (Name == "log")    return std::log(arg(0));
  if (Name == "log10")  return std::log10(arg(0));
  if (Name == "log2")   return std::log2(arg(0));
  if (Name == "sqrt")   return std::sqrt(arg(0));
  if (Name == "abs")    return std::fabs(arg(0));
  if (Name == "sign") { double V = arg(0); return (V > 0) - (V < 0); }
  if (Name == "floor")  return std::floor(arg(0));
  if (Name == "ceil")   return std::ceil(arg(0));
  if (Name == "round")  return std::round(arg(0));
  if (Name == "min")    return std::fmin(arg(0), arg(1));
  if (Name == "max")    return std::fmax(arg(0), arg(1));
  if (Name == "mod")    return std::fmod(arg(0), arg(1));
  if (Name == "rem")    return arg(0) - arg(1) * std::trunc(arg(0) / arg(1));
  if (Name == "pow")    return std::pow(arg(0), arg(1));
  if (Name == "hypot")  return std::hypot(arg(0), arg(1));
  if (Name == "square") { double V = arg(0); return V * V; }
  return 0.0;
}

double interpExpr(const matlab::Expr *E, InterpEnv &Env) {
  if (!E) return 0.0;
  // Local alias to dodge the `NodeKind` already in scope from the
  // expression-evaluator namespace block (typedef'd to
  // `MatlabFcnTree::K`). Fully qualifying disambiguates.
  using MNK = matlab::NodeKind;
  switch (E->Kind) {
  case MNK::IntegerLiteral: {
    auto *IL = static_cast<const matlab::IntegerLiteral *>(E);
    try { return std::stod(std::string(IL->Text)); } catch (...) { return 0.0; }
  }
  case MNK::FPLiteral: {
    auto *FL = static_cast<const matlab::FPLiteral *>(E);
    try { return std::stod(std::string(FL->Text)); } catch (...) { return 0.0; }
  }
  case MNK::NameExpr: {
    auto *NE = static_cast<const matlab::NameExpr *>(E);
    std::string Name(NE->Name);
    if (Name == "pi") return M_PI;
    if (Name == "e")  return M_E;
    auto It = Env.Vars.find(Name);
    return It == Env.Vars.end() ? 0.0 : It->second;
  }
  case MNK::UnaryOp: {
    auto *U = static_cast<const matlab::UnaryOpExpr *>(E);
    double V = interpExpr(U->Operand, Env);
    switch (U->Op) {
    case matlab::UnOp::Plus:  return V;
    case matlab::UnOp::Minus: return -V;
    case matlab::UnOp::Not:   return V == 0.0 ? 1.0 : 0.0;
    }
    return V;
  }
  case MNK::BinaryOp: {
    auto *B = static_cast<const matlab::BinaryOpExpr *>(E);
    double L = interpExpr(B->LHS, Env);
    double R = interpExpr(B->RHS, Env);
    using MO = matlab::BinOp;
    switch (B->Op) {
    case MO::Add: return L + R;
    case MO::Sub: return L - R;
    case MO::Mul: case MO::ElemMul: return L * R;
    case MO::Div: case MO::ElemDiv: return L / R;
    case MO::LeftDiv: case MO::ElemLeftDiv: return R / L;
    case MO::Pow: case MO::ElemPow: return std::pow(L, R);
    case MO::Eq: return L == R ? 1.0 : 0.0;
    case MO::Ne: return L != R ? 1.0 : 0.0;
    case MO::Lt: return L <  R ? 1.0 : 0.0;
    case MO::Le: return L <= R ? 1.0 : 0.0;
    case MO::Gt: return L >  R ? 1.0 : 0.0;
    case MO::Ge: return L >= R ? 1.0 : 0.0;
    case MO::And: case MO::ShortAnd:
      return (L != 0.0 && R != 0.0) ? 1.0 : 0.0;
    case MO::Or: case MO::ShortOr:
      return (L != 0.0 || R != 0.0) ? 1.0 : 0.0;
    }
    return 0.0;
  }
  case MNK::CallOrIndex: {
    auto *CI = static_cast<const matlab::CallOrIndex *>(E);
    if (!CI->Callee || CI->Callee->Kind != MNK::NameExpr) return 0.0;
    auto *Callee = static_cast<const matlab::NameExpr *>(CI->Callee);
    std::vector<double> Args;
    Args.reserve(CI->Args.size());
    for (auto *A : CI->Args) Args.push_back(interpExpr(A, Env));
    return callBuiltinScalar(std::string(Callee->Name), Args);
  }
  default:
    return 0.0;
  }
}

void interpStmt(const matlab::Stmt *S, InterpEnv &Env);

void interpBlock(const matlab::Block *B, InterpEnv &Env) {
  if (!B) return;
  for (auto *S : B->Stmts) interpStmt(S, Env);
}

void interpStmt(const matlab::Stmt *S, InterpEnv &Env) {
  using MNK = matlab::NodeKind;
  if (!S) return;
  switch (S->Kind) {
  case MNK::ExprStmt: {
    auto *ES = static_cast<const matlab::ExprStmt *>(S);
    interpExpr(ES->E, Env);
    return;
  }
  case MNK::AssignStmt: {
    auto *AS = static_cast<const matlab::AssignStmt *>(S);
    if (AS->LHS.size() != 1) return;
    auto *Lhs = AS->LHS.front();
    if (!Lhs || Lhs->Kind != MNK::NameExpr) return;
    auto *NE = static_cast<const matlab::NameExpr *>(Lhs);
    Env.Vars[std::string(NE->Name)] = interpExpr(AS->RHS, Env);
    return;
  }
  case MNK::IfStmt: {
    auto *I = static_cast<const matlab::IfStmt *>(S);
    if (interpExpr(I->Cond, Env) != 0.0) {
      interpBlock(I->Then, Env);
      return;
    }
    for (auto &EI : I->Elseifs) {
      if (interpExpr(EI.Cond, Env) != 0.0) {
        interpBlock(EI.Body, Env);
        return;
      }
    }
    if (I->Else) interpBlock(I->Else, Env);
    return;
  }
  case MNK::Block:
    interpBlock(static_cast<const matlab::Block *>(S), Env);
    return;
  case MNK::ReturnStmt:
    throw ReturnSignal{};
  default:
    return;
  }
}

double runMatlabFunction(const MflowLinkSim::MatlabFunctionState &S,
                         const std::vector<double> &Inputs,
                         double T) {
  InterpEnv Env;
  Env.Vars["t"] = T;
  for (size_t I = 0; I < S.InNames.size(); ++I)
    Env.Vars[S.InNames[I]] = I < Inputs.size() ? Inputs[I] : 0.0;
  // The shorthand `u` alias (matches our expression-evaluator
  // semantics — first input is `u`).
  if (!Inputs.empty()) Env.Vars["u"] = Inputs.front();
  try {
    interpBlock(S.Fn->Body, Env);
  } catch (const ReturnSignal &) {
    // Normal `return` exit.
  }
  auto It = Env.Vars.find(S.OutName);
  return It == Env.Vars.end() ? 0.0 : It->second;
}

} // namespace

std::string validateMatlabFunctionBody(const std::string &Source) {
  auto Pair = parseMatlabFunctionBody(Source);
  if (Pair.first) return std::string{};
  return Pair.second.empty() ? std::string("invalid function_body")
                              : Pair.second;
}

} // namespace matlab::flowchart
