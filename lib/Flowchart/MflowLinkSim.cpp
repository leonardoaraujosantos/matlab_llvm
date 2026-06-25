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

// §17.5 #8 — process-wide JIT factory snapshot. Updated by
// `setMatlabFcnJit`; captured at sim construction so a later install
// can't change semantics mid-run. Initialised to all-null = no JIT,
// AST interpreter always wins.
MatlabFcnJit &globalJitSlot() {
  static MatlabFcnJit S{};
  return S;
}

} // namespace

void setMatlabFcnJit(MatlabFcnJit Jit) { globalJitSlot() = Jit; }

const MatlabFcnJit &currentMatlabFcnJit() { return globalJitSlot(); }

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

// mflow-3d-animation — parse a `"x,y,z"` (or space/`;`-separated) vector param
// into `Out[0..2]`. Missing param leaves `Out` untouched (so the caller's
// default survives). Fewer than three values fill from the front; extras are
// ignored. Brackets are tolerated.
void parseVec3Param(const MflBlock &B, const char *Key, double Out[3]) {
  const std::string *S = paramS(B, Key);
  if (!S) return;
  int N = 0;
  std::string Tok;
  std::string Str = *S;
  Str.push_back(',');
  for (char C : Str) {
    if (C == ',' || C == ' ' || C == '\t' || C == ';' || C == '[' ||
        C == ']') {
      if (!Tok.empty()) {
        if (N < 3) {
          try { Out[N] = std::stod(Tok); } catch (...) {}
          ++N;
        }
        Tok.clear();
      }
    } else {
      Tok.push_back(C);
    }
  }
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
// Parse a MATLAB matrix literal like "1 0; -1 0" / "[2; 0]" into a flat
// row-major value list + row/col counts. Whitespace OR commas separate
// columns; `;` separates rows. Shared by the state_space evaluator and the
// vector-x0 initial condition (#345).
void parseSimMatrix(const std::string &S, std::vector<double> &Vals,
                    int &Rows, int &Cols) {
  Vals.clear();
  Rows = 0;
  Cols = 0;
  std::string T = S;
  auto F = T.find('[');
  if (F != std::string::npos) T.erase(0, F + 1);
  auto L = T.rfind(']');
  if (L != std::string::npos) T.erase(L);
  for (char &c : T) if (c == ',') c = ' ';
  std::stringstream RS(T);
  std::string Row;
  while (std::getline(RS, Row, ';')) {
    int C = 0;
    std::stringstream CS(Row);
    std::string Tok;
    while (CS >> Tok) {
      try { Vals.push_back(std::stod(Tok)); } catch (...) {}
      ++C;
    }
    if (C > 0) { ++Rows; Cols = C; }
  }
}

//===----------------------------------------------------------------------===//
// Small dense linear algebra for the Kalman-filter block (#343). Matrices are
// flat row-major `vector<double>` with explicit (rows, cols). These are sized
// by the user's A/C/Q/R literals (typically ≤ a handful of states), so a plain
// O(n³) implementation is more than adequate.
//===----------------------------------------------------------------------===//

// #343 — the discrete-IIR family that shares the direct-form-II difference
// engine (TFCache_ / Z_ / FirHistory_ / fireDiscreteTicks). discrete_filter
// takes num/den directly; biquad and the streaming filters build num/den from
// their own parameters at cache time.
static bool isDiscreteIirKind(const std::string &K) {
  return K == "signal_discrete_filter" || K == "signal_biquad" ||
         K == "signal_lowpass" || K == "signal_highpass" ||
         K == "signal_dcblock";
}

// #343 Vision — named 3×3 image-filter kernels (row-major). Returns false for
// an unknown name so the caller can fall back to an explicit `kernel` literal.
static bool namedKernel(const std::string &Name, std::vector<double> &K,
                        int &Kr, int &Kc) {
  Kr = 3; Kc = 3;
  if (Name == "box") {
    K.assign(9, 1.0 / 9.0);
  } else if (Name == "gaussian3") {
    K = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    for (auto &v : K) v /= 16.0;
  } else if (Name == "sobelx") {
    K = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  } else if (Name == "sobely") {
    K = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  } else {
    return false;
  }
  return true;
}

// DSP window taper coefficient at index k of an N-point window (#343).
// Supported: rectangular ("rect"/"none"), Hamming, Blackman; default Hann.
static double windowCoef(const std::string *Type, int k, int N) {
  if (N <= 1) return 1.0;
  double t = 2.0 * M_PI * k / (N - 1);
  const std::string ty = Type ? *Type : "hann";
  if (ty == "rect" || ty == "rectangular" || ty == "none") return 1.0;
  if (ty == "hamming") return 0.54 - 0.46 * std::cos(t);
  if (ty == "blackman")
    return 0.42 - 0.5 * std::cos(t) + 0.08 * std::cos(2.0 * t);
  return 0.5 * (1.0 - std::cos(t)); // Hann (default)
}

// C[r×c] = A[r×k] · B[k×c].
static std::vector<double> matMul(const std::vector<double> &A, int Ar, int Ak,
                                  const std::vector<double> &B, int Bk, int Bc) {
  std::vector<double> Cm(static_cast<size_t>(Ar) * Bc, 0.0);
  if (Ak != Bk) return Cm; // non-conformant → zeros (caller validates)
  for (int i = 0; i < Ar; ++i)
    for (int j = 0; j < Bc; ++j) {
      double s = 0.0;
      for (int k = 0; k < Ak; ++k)
        s += A[static_cast<size_t>(i) * Ak + k] * B[static_cast<size_t>(k) * Bc + j];
      Cm[static_cast<size_t>(i) * Bc + j] = s;
    }
  return Cm;
}

// Aᵀ for an r×c matrix.
static std::vector<double> matT(const std::vector<double> &A, int Ar, int Ac) {
  std::vector<double> T(static_cast<size_t>(Ar) * Ac, 0.0);
  for (int i = 0; i < Ar; ++i)
    for (int j = 0; j < Ac; ++j)
      T[static_cast<size_t>(j) * Ar + i] = A[static_cast<size_t>(i) * Ac + j];
  return T;
}

// In-place A += B (both r×c). Used for P⁻ = A·P·Aᵀ + Q etc.
static void matAddInto(std::vector<double> &A, const std::vector<double> &B) {
  for (size_t i = 0; i < A.size() && i < B.size(); ++i) A[i] += B[i];
}

// Inverse of an m×m matrix via Gauss-Jordan with partial pivoting. Returns
// false (and leaves Out untouched) if the matrix is singular to working tol.
static bool matInv(const std::vector<double> &A, int m,
                   std::vector<double> &Out) {
  std::vector<double> M = A; // working copy
  Out.assign(static_cast<size_t>(m) * m, 0.0);
  for (int i = 0; i < m; ++i) Out[static_cast<size_t>(i) * m + i] = 1.0;
  for (int col = 0; col < m; ++col) {
    // Partial pivot: largest |entry| in this column at/below the diagonal.
    int piv = col;
    double best = std::fabs(M[static_cast<size_t>(col) * m + col]);
    for (int r = col + 1; r < m; ++r) {
      double v = std::fabs(M[static_cast<size_t>(r) * m + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-12) return false; // singular
    if (piv != col)
      for (int j = 0; j < m; ++j) {
        std::swap(M[static_cast<size_t>(col) * m + j],
                  M[static_cast<size_t>(piv) * m + j]);
        std::swap(Out[static_cast<size_t>(col) * m + j],
                  Out[static_cast<size_t>(piv) * m + j]);
      }
    double d = M[static_cast<size_t>(col) * m + col];
    for (int j = 0; j < m; ++j) {
      M[static_cast<size_t>(col) * m + j] /= d;
      Out[static_cast<size_t>(col) * m + j] /= d;
    }
    for (int r = 0; r < m; ++r) {
      if (r == col) continue;
      double f = M[static_cast<size_t>(r) * m + col];
      if (f == 0.0) continue;
      for (int j = 0; j < m; ++j) {
        M[static_cast<size_t>(r) * m + j] -= f * M[static_cast<size_t>(col) * m + j];
        Out[static_cast<size_t>(r) * m + j] -= f * Out[static_cast<size_t>(col) * m + j];
      }
    }
  }
  return true;
}

std::pair<std::unique_ptr<MflowLinkSim::MatlabFunctionState>, std::string>
parseMatlabFunctionBody(const std::string &Source);
std::vector<double> runMatlabFunction(const MflowLinkSim::MatlabFunctionState &S,
                                      const std::vector<double> &Inputs, double T,
                                      const std::set<int> *WatchLines = nullptr,
                                      int *HitLine = nullptr,
                                      std::map<std::string, double> *HitVars = nullptr,
                                      int StopAtStmt = -1, int *HitStmtIdx = nullptr);
// §17.5 #8 — accessor for the parsed-function input count. The
// constructor needs this to decide how many `u1..uN` slots the JIT
// wrapper should declare, but `MatlabFunctionState` is defined
// further down (after AST/Lexer/Parser have established their full
// types). Defined alongside the struct definition itself.
unsigned matlabFunctionInputCount(
    const MflowLinkSim::MatlabFunctionState &S);
unsigned matlabFunctionOutputCount(
    const MflowLinkSim::MatlabFunctionState &S);
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
  OutRows_.resize(N, 1);
  OutCols_.resize(N, 1);
  OutShape_.assign(N, {});
  VecOut_.assign(N, {});
  PortOut_.assign(N, {});
  for (size_t I = 0; I < N; ++I) {
    int W = M_.Blocks[I].OutWidth;
    if (W < 1) W = 1;
    OutWidth_[I] = W;
    // mflow-nd-signals — the canonical shape is `OutShape` (rank 1–6) when the
    // lowering stamped one and its product matches the width; otherwise derive
    // a 2-D shape from OutRows/OutCols (§17.5 #9), else fall back to (1 × W).
    std::vector<int> Sh = M_.Blocks[I].OutShape;
    int prod = 1;
    for (int d : Sh) prod *= (d > 0 ? d : 0);
    if (Sh.empty() || prod != W) {
      int R = M_.Blocks[I].OutRows;
      int C = M_.Blocks[I].OutCols;
      if (R <= 0 || C <= 0 || R * C != W) { R = 1; C = W; }
      Sh = (R > 1) ? std::vector<int>{R, C} : std::vector<int>{W};
    }
    OutShape_[I] = Sh;
    // OutRows/OutCols stay the 2-D projection so every legacy 1-D/2-D site
    // reads identical values: rank-1 → (1 × W); rank-≥2 → (dim0, prod(rest)).
    if (Sh.size() <= 1) {
      OutRows_[I] = 1;
      OutCols_[I] = W;
    } else {
      OutRows_[I] = Sh[0];
      int rest = 1;
      for (size_t d = 1; d < Sh.size(); ++d) rest *= Sh[d];
      OutCols_[I] = rest;
    }
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
  FirHistory_.assign(N, {});
  // Pre-size FIR history per discrete_filter block to max(NumLen,
  // DenLen) so every reference index is in-range.
  for (size_t I = 0; I < N; ++I) {
    if (!isDiscreteIirKind(M_.Blocks[I].Kind))
      continue;
    const auto &TF = TFCache_[I];
    size_t Sz = std::max(TF.Num.size(), TF.Den.size());
    if (Sz == 0) Sz = 1;
    FirHistory_[I].assign(Sz, 0.0);
  }

  // Input wiring. Sum / Product blocks read in1, in2, …; every other
  // single-input block reads "in".
  for (auto &E : M_.Edges) {
    auto FI = IdxOf.find(E.FromBlock);
    auto TI = IdxOf.find(E.ToBlock);
    if (FI == IdxOf.end() || TI == IdxOf.end()) continue;
    Inputs_[TI->second].push_back({FI->second, E.FromPort, E.ToPort});
  }

  // Cache transfer-function coefficients. For `signal_zero_pole`,
  // we run the same path after expanding zeros/poles into num/den
  // polynomials — the evaluator reuses the transfer_fcn machinery
  // verbatim from then on.
  TransportBuf_.assign(N, {});
  Lookup1DCache_.assign(N, {});
  Lookup2DCache_.assign(N, {});
  LookupNDCache_.assign(N, {});
  NoiseSeed_.assign(N, 0);
  DigitalLatch_.assign(N, 0.0);
  HdlMem_.assign(N, {});
  ErrAccum_.assign(N, 0.0);
  TotAccum_.assign(N, 0.0);
  RunCount_.assign(N, 0.0);
  RunMean_.assign(N, 0.0);
  RunM2_.assign(N, 0.0);
  Kalman_.assign(N, {});
  MatlabFcnCache_.resize(N);
  MatlabFnCache_.resize(N);
  // §17.5 #8 — snapshot the currently installed JIT factory once.
  // Subsequent installs only affect later simulators.
  MatlabFcnJitOps_ = currentMatlabFcnJit();
  MatlabFnJit_.assign(N, nullptr);
  MatlabFnJitArity_.assign(N, 0);
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
    } else if (B.Kind == "signal_biquad") {
      // #343 DSP — second-order section:
      //   H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (a0 + a1 z⁻¹ + a2 z⁻²)
      // Coefficients via `b`/`a` vector strings ("b0 b1 b2") or the
      // individual b0..b2 / a0..a2 scalars (Simulink Biquad style). The
      // generic discrete_filter evaluator then runs the difference equation.
      // parseSimMatrix tolerates space- OR comma-separated coefficients
      // (parsePoly is comma-only), so "0.07 0.13 0.07" and "0.07,0.13,0.07"
      // both work.
      std::vector<double> Num, Den;
      int pr = 0, pc = 0;
      if (auto *BS = paramS(B, "b")) parseSimMatrix(*BS, Num, pr, pc);
      else Num = {MflowLinkSim::paramD(B, "b0", 1.0),
                  MflowLinkSim::paramD(B, "b1", 0.0),
                  MflowLinkSim::paramD(B, "b2", 0.0)};
      if (auto *AS = paramS(B, "a")) parseSimMatrix(*AS, Den, pr, pc);
      else Den = {MflowLinkSim::paramD(B, "a0", 1.0),
                  MflowLinkSim::paramD(B, "a1", 0.0),
                  MflowLinkSim::paramD(B, "a2", 0.0)};
      // A 2nd-order section always carries 2 state slots — pad to length 3.
      Num.resize(3, 0.0);
      Den.resize(3, 0.0);
      if (Den[0] == 0.0) Den[0] = 1.0; // normalise a0
      TFCache_[I].Num = std::move(Num);
      TFCache_[I].Den = std::move(Den);
      TFCache_[I].Valid = true;
    } else if (B.Kind == "signal_lowpass" || B.Kind == "signal_highpass" ||
               B.Kind == "signal_dcblock") {
      // #343 DSP — streaming first-order filters as discrete_filter presets.
      // `alpha`∈(0,1] sets the pole; `lowpass` defaults to heavy smoothing,
      // `highpass`/`dcblock` to a pole near 1 (wide passband).
      std::vector<double> Num, Den;
      if (B.Kind == "signal_lowpass") {
        // y[n] = α·x[n] + (1-α)·y[n-1] — H(z)=α/(1-(1-α)z⁻¹), unity DC gain.
        double A = MflowLinkSim::paramD(B, "alpha", 0.1);
        Num = {A};
        Den = {1.0, -(1.0 - A)};
      } else if (B.Kind == "signal_highpass") {
        // H(z) = α(1 - z⁻¹)/(1 - α z⁻¹) — zero DC gain, ~unity at Nyquist.
        double A = MflowLinkSim::paramD(B, "alpha", 0.9);
        Num = {A, -A};
        Den = {1.0, -A};
      } else { // signal_dcblock
        // H(z) = (1 - z⁻¹)/(1 - r z⁻¹) — removes DC, passes everything else.
        double R = MflowLinkSim::paramD(B, "r", 0.995);
        Num = {1.0, -1.0};
        Den = {1.0, -R};
      }
      TFCache_[I].Num = std::move(Num);
      TFCache_[I].Den = std::move(Den);
      TFCache_[I].Valid = true;
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
    } else if (B.Kind == "signal_lookup_nd") {
      // N-D table: `breakpoints1..breakpointsN` (one per dimension, space/comma
      // separated) + a flat row-major `tableData` (dim 0 outermost). N is the
      // number of breakpoints params present (≤ 6).
      auto &C = LookupNDCache_[I];
      for (int d = 1; d <= 6; ++d) {
        auto *S = paramS(B, ("breakpoints" + std::to_string(d)).c_str());
        if (!S) break;
        std::vector<double> Ax; int r = 0, c = 0;
        parseSimMatrix(*S, Ax, r, c);
        if (Ax.empty()) break;
        C.Axes.push_back(std::move(Ax));
      }
      if (auto *T = paramS(B, "tableData")) {
        std::vector<double> Z; int r = 0, c = 0;
        parseSimMatrix(*T, Z, r, c);
        C.Z = std::move(Z);
      }
      size_t Prod = C.Axes.empty() ? 0 : 1;
      for (auto &Ax : C.Axes) Prod *= Ax.size();
      C.Valid = !C.Axes.empty() && Prod > 0 && Prod == C.Z.size();
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
        if (Pair.first) {
          // §17.5 #8 — try the JIT factory first when installed.
          // The AST interpreter parse above is what we fall back to
          // (and what `MatlabFnCache_[I]` keeps for the fallback
          // path); we use its signature to know NumInputs / OutName.
          if (MatlabFcnJitOps_.Compile && MatlabFcnJitOps_.Call &&
              MatlabFcnJitOps_.Release) {
            unsigned NumInputs = matlabFunctionInputCount(*Pair.first);
            // Cap at 8 to match the wrapper's pre-generated signature
            // pad — bodies with more inputs fall back to the AST
            // interpreter rather than failing the sim. #344: the JIT
            // wrapper returns a single scalar, so multi-output bodies
            // also fall back to the AST interpreter (which fills every
            // out1..outM port).
            if (NumInputs <= 8 &&
                matlabFunctionOutputCount(*Pair.first) == 1) {
              std::string JitErr;
              if (auto *H = MatlabFcnJitOps_.Compile(*FB, NumInputs,
                                                     JitErr)) {
                MatlabFnJit_[I]      = H;
                MatlabFnJitArity_[I] = NumInputs;
              }
              // On JIT failure we silently keep the AST fallback —
              // the user's signal-flow keeps running; advanced
              // language constructs that the interpreter doesn't
              // implement will just return 0 until they're added.
              // The `JitErr` text is intentionally not surfaced
              // here to avoid swamping the user with diagnostics
              // when the JIT factory is opt-in.
              (void)JitErr;
            }
          }
          MatlabFnCache_[I] = std::move(Pair.first);
        }
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
  // §17.5 #7 — per-block solver overrides. Each MflBlock may carry
  // a non-NaN MaxStepOverride from its enclosing flow's solver.
  // The global step is the tightest cap: `min` over all overrides
  // + the global MaxStep. Tolerances similarly tighten globally.
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    if (!std::isnan(B.MaxStepOverride) && B.MaxStepOverride > 0.0 &&
        B.MaxStepOverride < StepSize_) {
      StepSize_ = B.MaxStepOverride;
    }
    if (!std::isnan(B.RelTolOverride) && B.RelTolOverride > 0.0 &&
        B.RelTolOverride < M_.Solver.RelTol) {
      // Mutate a local copy of the solver config doesn't apply here
      // since M_.Solver is const; instead the runtime picks the
      // tightest tolerance through this read at substep time.
      // Stash the global tightening once.
    }
  }
  // Recompute effective relTol / absTol as the tightest demanded.
  EffectiveRelTol_ = M_.Solver.RelTol;
  EffectiveAbsTol_ = M_.Solver.AbsTol;
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    if (!std::isnan(B.RelTolOverride) && B.RelTolOverride > 0.0 &&
        B.RelTolOverride < EffectiveRelTol_)
      EffectiveRelTol_ = B.RelTolOverride;
    if (!std::isnan(B.AbsTolOverride) && B.AbsTolOverride > 0.0 &&
        B.AbsTolOverride < EffectiveAbsTol_)
      EffectiveAbsTol_ = B.AbsTolOverride;
  }
  // Item-2 — adaptive step is enabled when the model asks for
  // variable_step + an adaptive algorithm. Fixed-step mode keeps
  // the legacy classic RK4 path. The ode15s lane (§17.5 #3) ships
  // as fixed-step implicit BDF1 — the user picks the step via
  // `settings.solver.maxStep`; BDF1 is L-stable so stiff systems
  // run at step sizes DOPRI5 wouldn't survive.
  AdaptiveSolver_ = (M_.Solver.Type != "fixed_step") &&
                    (M_.Solver.Algorithm == "ode45" ||
                     M_.Solver.Algorithm == "ode23");
  // mflow-variable-step-stiff-solvers — `ode23` runs the native
  // Bogacki-Shampine 3(2) pair (embedded 2nd-order estimate); `ode45`
  // runs Dormand-Prince 5(4) (embedded 4th-order). The controller
  // exponent is keyed off the estimate's order via AdaptiveErrOrder_.
  if (M_.Solver.Algorithm == "ode23") {
    AdaptiveMethod_ = AdaptiveMethod::BS32;
    AdaptiveErrOrder_ = 2;
  } else {
    AdaptiveMethod_ = AdaptiveMethod::DOPRI5;
    AdaptiveErrOrder_ = 4;
  }
  // mflow-variable-step-stiff-solvers — the implicit/stiff lane covers
  // `ode15s` (BDF1, Newton) and `ode23s` (modified Rosenbrock). Selecting
  // either routes away from the explicit RK4 fall-through.
  Implicit_ = (M_.Solver.Algorithm == "ode15s" ||
               M_.Solver.Algorithm == "ode23s" ||
               M_.Solver.Algorithm == "ode23t" ||
               M_.Solver.Algorithm == "ode23tb");
  if (M_.Solver.Algorithm == "ode23s")
    StiffMethod_ = StiffMethod::ROSENBROCK;
  else if (M_.Solver.Algorithm == "ode23t")
    StiffMethod_ = StiffMethod::TRAPEZOIDAL;
  else if (M_.Solver.Algorithm == "ode23tb")
    StiffMethod_ = StiffMethod::TRBDF2;
  else
    StiffMethod_ = StiffMethod::BDF1;
  // Adaptive stiff stepping is wired for BDF1 (`ode15s`) under variable_step;
  // the other stiff methods remain fixed-step for now.
  ImplicitAdaptive_ = Implicit_ && (M_.Solver.Type != "fixed_step") &&
                      (StiffMethod_ == StiffMethod::BDF1);
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
                    B.Kind == "signal_scope3d" ||
                    B.Kind == "signal_actor3d" ||
                    B.Kind == "signal_display" ||
                    B.Kind == "signal_to_workspace";
    if (!B.LogSignal && !Implicit) continue;
    std::string Name = B.Id;
    if (B.Kind == "signal_to_workspace") {
      if (auto *V = paramS(B, "variableName")) Name = *V;
    }
    // 3-D scope — one column per axis, labelled `<id>[x]/[y]/[z]`, so a
    // trajectory viewer can render the (x, y, z) path.
    if (B.Kind == "signal_scope3d") {
      static const char *Axes[3] = {"x", "y", "z"};
      for (int E = 0; E < 3; ++E) {
        LogBlocks_.push_back(I);
        LogElements_.push_back(E);
        LogNames_.push_back(Name + "[" + Axes[E] + "]");
      }
      continue;
    }
    // mflow-3d-animation — actor transform timeline: nine columns
    // `<id>[tx,ty,tz,rx,ry,rz,sx,sy,sz]` that the Babylon emit lane reads as
    // the per-step animation keyframes (translation, roll/pitch/yaw, scale).
    if (B.Kind == "signal_actor3d") {
      static const char *Comp[9] = {"tx", "ty", "tz", "rx", "ry",
                                    "rz", "sx", "sy", "sz"};
      for (int E = 0; E < 9; ++E) {
        LogBlocks_.push_back(I);
        LogElements_.push_back(E);
        LogNames_.push_back(Name + "[" + Comp[E] + "]");
      }
      // URDF actor (Tier 3b) — additionally log up to 12 joint angles.
      if (OutWidth_[I] > 9) {
        for (int Q = 0; Q < OutWidth_[I] - 9; ++Q) {
          LogBlocks_.push_back(I);
          LogElements_.push_back(9 + Q);
          LogNames_.push_back(Name + "[q" + std::to_string(Q + 1) + "]");
        }
      }
      continue;
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
      // mflow-nd-signals — one CSV column per element. For a rank-N signal
      // (N ≥ 2) emit `<id>[i1,…,iN]` (1-based, row-major); a 1-D vector keeps
      // the legacy `<id>[k]` form so existing IDE consumers and tests stay
      // byte-identical. The 2-D case renders exactly `<id>[r,c]` as before.
      const std::vector<int> &Sh = OutShape_[I];
      // Render multi-axis indices only when ≥2 dimensions exceed 1 — a vector
      // (row/column, only one non-singleton axis) keeps the legacy `[k]` form,
      // matching the pre-N-D 2-D rule (`R > 1 && C > 1`).
      int nonUnit = 0;
      for (int d : Sh) if (d > 1) ++nonUnit;
      bool MultiD = (nonUnit >= 2 && W > 1);
      for (int E = 0; E < W; ++E) {
        LogBlocks_.push_back(I);
        LogElements_.push_back(E);
        if (MultiD) {
          // Row-major de-linearization: rightmost dim varies fastest.
          std::string Idx;
          int rem = E;
          std::vector<int> sub(Sh.size());
          for (int d = (int)Sh.size() - 1; d >= 0; --d) {
            int dim = Sh[d] > 0 ? Sh[d] : 1;
            sub[d] = rem % dim;
            rem /= dim;
          }
          for (size_t d = 0; d < sub.size(); ++d) {
            if (d) Idx += ",";
            Idx += std::to_string(sub[d] + 1);
          }
          LogNames_.push_back(Name + "[" + Idx + "]");
        } else {
          LogNames_.push_back(Name + "[" + std::to_string(E + 1) + "]");
        }
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
  // #354/#386 — a restart clears any pending breakpoint hit + stepping session.
  LastSourceHit_ = SourceHit{};
  PendingStep_ = FcnStepState{};
  FcnStep_ = FcnStepState{};
  std::fill(Y_.begin(), Y_.end(), 0.0);
  std::fill(Z_.begin(), Z_.end(), 0.0);
  std::fill(Znext_.begin(), Znext_.end(), 0.0);
  std::fill(DiscPrevU_.begin(), DiscPrevU_.end(), 0.0);
  for (auto &H : FirHistory_) std::fill(H.begin(), H.end(), 0.0);
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
    } else if (B.Kind == "signal_pid") {
      size_t Off = StateOffset_[I];
      Y_[Off + 0] = paramD(B, "initialIntegral", 0.0);
      Y_[Off + 1] = 0.0; // derivative filter starts at rest
    } else if (B.Kind == "signal_state_space") {
      // #345: x0 may be a per-state vector literal ("2; 0" → state 1 = 2,
      // state 2 = 0). Parse it as a matrix and assign element-wise; a single
      // scalar broadcasts to every state (the old behaviour). A short vector
      // zero-fills the remaining states.
      std::vector<double> X0v;
      int X0r = 0, X0c = 0;
      if (const std::string *X0S = paramS(B, "x0"))
        parseSimMatrix(*X0S, X0v, X0r, X0c);
      for (int J = 0; J < B.ContStateCount; ++J) {
        double V;
        if (X0v.empty())          V = 0.0;
        else if (X0v.size() == 1) V = X0v[0];                 // scalar broadcast
        else V = (J < static_cast<int>(X0v.size())) ? X0v[J] : 0.0;
        Y_[StateOffset_[I] + J] = V;
      }
    } else if (B.Kind == "signal_kalman") {
      // #343 — discrete Kalman filter. Parse A/C/Q/R (+ optional B) once and
      // size the estimate X (length N = rows of A) and covariance Pc (N×N).
      // Dimensions must conform or the block degrades to a pass-through.
      KalmanState &KS = Kalman_[I];
      int Ar = 0, Ac = 0, Cr = 0, Cc = 0;
      if (const std::string *S = paramS(B, "A")) parseSimMatrix(*S, KS.A, Ar, Ac);
      if (const std::string *S = paramS(B, "C")) parseSimMatrix(*S, KS.C, Cr, Cc);
      KS.N = Ar;
      KS.Mz = Cr;
      KS.Valid = (Ar > 0 && Ar == Ac && Cc == Ar && Cr > 0);
      // Q defaults to 0 (no process noise), R to identity (unit measurement
      // noise) when unset or malformed; both must be square of the right size.
      int Qr = 0, Qc = 0, Rr = 0, Rc = 0;
      if (const std::string *S = paramS(B, "Q")) parseSimMatrix(*S, KS.Q, Qr, Qc);
      if (Qr != KS.N || Qc != KS.N)
        KS.Q.assign(static_cast<size_t>(KS.N) * KS.N, 0.0);
      if (const std::string *S = paramS(B, "R")) parseSimMatrix(*S, KS.R, Rr, Rc);
      if (Rr != KS.Mz || Rc != KS.Mz) {
        KS.R.assign(static_cast<size_t>(KS.Mz) * KS.Mz, 0.0);
        for (int d = 0; d < KS.Mz; ++d)
          KS.R[static_cast<size_t>(d) * KS.Mz + d] = 1.0;
      }
      // Optional control matrix B (N×P).
      int Br = 0, Bc = 0;
      if (const std::string *S = paramS(B, "B")) parseSimMatrix(*S, KS.B, Br, Bc);
      KS.P = (Br == KS.N && Bc > 0) ? Bc : 0;
      if (KS.P == 0) KS.B.clear();
      // Initial estimate x0 (vector, default 0); scalar broadcasts.
      std::vector<double> X0v; int X0r = 0, X0c = 0;
      if (const std::string *S = paramS(B, "x0")) parseSimMatrix(*S, X0v, X0r, X0c);
      KS.X.assign(KS.N, 0.0);
      for (int J = 0; J < KS.N; ++J)
        KS.X[J] = X0v.empty() ? 0.0
                  : (X0v.size() == 1 ? X0v[0]
                                     : (J < (int)X0v.size() ? X0v[J] : 0.0));
      // Initial covariance P0: full N×N literal, scalar (×I), or default I.
      std::vector<double> P0v; int P0r = 0, P0c = 0;
      if (const std::string *S = paramS(B, "P0")) parseSimMatrix(*S, P0v, P0r, P0c);
      KS.Pc.assign(static_cast<size_t>(KS.N) * KS.N, 0.0);
      if (P0r == KS.N && P0c == KS.N) {
        KS.Pc = P0v;
      } else {
        double diag = P0v.size() == 1 ? P0v[0] : 1.0;
        for (int d = 0; d < KS.N; ++d)
          KS.Pc[static_cast<size_t>(d) * KS.N + d] = diag;
      }
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
               B.Kind == "signal_discrete_filter" ||
               B.Kind == "signal_biquad" ||
               B.Kind == "signal_lowpass" || B.Kind == "signal_highpass" ||
               B.Kind == "signal_dcblock") {
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
    } else if (B.Kind == "signal_noise" || B.Kind == "signal_awgn") {
      // Per-block xorshift seed. The default seed makes the same
      // model reproducible across runs; users can override via
      // `params.seed`. signal_awgn (#343) shares the same RNG state.
      uint64_t Seed =
          static_cast<uint64_t>(paramD(B, "seed", 1.0));
      if (Seed == 0) Seed = 0xC0FFEE12345678ABULL;
      NoiseSeed_[I] = Seed;
    } else if (B.Kind == "signal_dff" || B.Kind == "signal_tff" ||
               B.Kind == "signal_counter" || B.Kind == "signal_jkff" ||
               B.Kind == "signal_srff") {
      // Clocked HDL registers start at their initial/reset value (#343).
      DigitalLatch_[I] = paramD(B, "initialValue", 0.0);
    } else if (B.Kind == "signal_shift_register") {
      // #343 — N-stage shift chain, all seeded to initialValue.
      int Len = std::max(1, (int)paramD(B, "length", 4.0));
      HdlMem_[I].assign(Len, paramD(B, "initialValue", 0.0));
    } else if (B.Kind == "signal_ram") {
      // #343 — depth-word RAM, seeded to initialValue (default 0).
      int Depth = std::max(1, (int)paramD(B, "depth", 8.0));
      HdlMem_[I].assign(Depth, paramD(B, "initialValue", 0.0));
    } else if (B.Kind == "signal_rom") {
      // #343 — read-only memory; `content` is a space/comma vector literal.
      std::vector<double> C;
      int r = 0, c = 0;
      if (const std::string *S = paramS(B, "content")) parseSimMatrix(*S, C, r, c);
      if (C.empty()) C.push_back(0.0);
      HdlMem_[I] = std::move(C);
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

  // Read the value flowing along an edge: a multi-output source publishes
  // its named ports in PortOut_, so a wire from `out2` reads that; every
  // other case falls back to the source block's scalar Out_ (#344 / #345).
  auto edgeValue = [&](const InputEdge &P) -> double {
    if (!P.SrcPort.empty()) {
      const auto &PM = PortOut_[P.SrcBlock];
      auto It = PM.find(P.SrcPort);
      if (It != PM.end()) return It->second;
    }
    return Out_[P.SrcBlock];
  };
  auto inputOf = [&](size_t I, const char *PortId) -> double {
    for (auto &P : Inputs_[I])
      if (P.DstPort == PortId) return edgeValue(P);
    return 0.0;
  };
  auto sumInput = [&](size_t I, const std::string &Port) -> double {
    // Multiple edges may land on the same input port — sum them
    // (matches Simulink's implicit summing convention for vector
    // joins, and harmless for the single-edge common case).
    double V = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == Port) V += edgeValue(P);
    return V;
  };
  // mflow-3d-animation — read element `Elem` of a (possibly vector) input
  // port. A width-1 source broadcasts its scalar to element 0 only; a vector
  // source (e.g. a Mux feeding a width-3 translation port) is read from its
  // VecOut_ slice. Returns NaN when the port has no edge, so the caller can
  // distinguish "unconnected" from "connected to 0".
  auto vecInput = [&](size_t I, const char *Port, int Elem) -> double {
    for (auto &P : Inputs_[I]) {
      if (P.DstPort != Port) continue;
      size_t S = P.SrcBlock;
      if (OutWidth_[S] > 1 && Elem < (int)VecOut_[S].size())
        return VecOut_[S][Elem];
      return Elem == 0 ? edgeValue(P) : 0.0;
    }
    return std::numeric_limits<double>::quiet_NaN();
  };
  auto portConnected = [&](size_t I, const char *Port) -> bool {
    for (auto &P : Inputs_[I])
      if (P.DstPort == Port) return true;
    return false;
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
    } else if (K == "signal_from_workspace") {
      // From Workspace — replay the inline `data` time-series ([t v; …]) at the
      // current sim time T. Linear interpolation by default; `interpolation:
      // "zoh"` holds. Clamped to the first/last sample outside the range. (Data
      // is small; parsed per-eval — the .mflow carries it, no live workspace.)
      std::vector<double> Tab;
      int R = 0, C = 0;
      if (const std::string *D = paramS(B, "data")) parseSimMatrix(*D, Tab, R, C);
      double V = 0.0;
      if (R >= 1 && C >= 2) {
        const std::string *Ip = paramS(B, "interpolation");
        bool ZOH = Ip && (*Ip == "zoh" || *Ip == "hold" || *Ip == "previous");
        auto tAt = [&](int r) { return Tab[(size_t)r * C]; };
        auto vAt = [&](int r) { return Tab[(size_t)r * C + 1]; };
        if (T <= tAt(0)) {
          V = vAt(0);
        } else if (T >= tAt(R - 1)) {
          V = vAt(R - 1);
        } else if (ZOH) {
          // Zero-order hold: the value of the last sample with time ≤ T (the
          // new sample takes effect at its own time).
          V = vAt(0);
          for (int r = 0; r < R; ++r)
            if (tAt(r) <= T) V = vAt(r); else break;
        } else {
          for (int r = 0; r + 1 < R; ++r) {
            if (T >= tAt(r) && T <= tAt(r + 1)) {
              double ta = tAt(r), tb = tAt(r + 1);
              V = (tb == ta)
                      ? vAt(r)
                      : vAt(r) + (vAt(r + 1) - vAt(r)) * (T - ta) / (tb - ta);
              break;
            }
          }
        }
      } else if (R * C >= 1) {
        V = Tab[0]; // a bare value list with no time column — hold the first
      }
      Out_[I] = V;
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
          for (auto &P : Inputs_[I]) Sum += edgeValue(P);
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
    } else if (K == "signal_pid") {
      // Continuous parallel-form PID with a first-order derivative filter:
      //   C(s) = Kp + Ki/s + Kd*N/(s+N)
      // States: x0 = ∫e dt, x1 = N/(s+N)·e (a low-pass of the error).
      //   x0' = e
      //   x1' = N*(e - x1)
      //   u   = Kp*e + Ki*x0 + Kd*N*(e - x1)
      // Optional output saturation (upperLimit/lowerLimit) with clamping
      // anti-windup: the integrator is frozen while the output is pinned to
      // a limit and the error would drive it further past that limit.
      size_t Off = StateOffset_[I];
      double E = sumInput(I, "in");
      double Kp = paramD(B, "Kp", 1.0);
      double Ki = paramD(B, "Ki", 0.0);
      double Kd = paramD(B, "Kd", 0.0);
      double Nf = paramD(B, "N", 100.0);
      if (Nf <= 0.0)
        Nf = 100.0;
      double Xi = State[Off + 0];
      double Xd = State[Off + 1];
      double U = Kp * E + Ki * Xi + Kd * Nf * (E - Xd);
      double Hi = paramD(B, "upperLimit", std::numeric_limits<double>::infinity());
      double Lo = paramD(B, "lowerLimit", -std::numeric_limits<double>::infinity());
      bool AtHi = U > Hi, AtLo = U < Lo;
      if (AtHi)
        U = Hi;
      else if (AtLo)
        U = Lo;
      Out_[I] = U;
      if (Deriv) {
        bool Freeze = (AtHi && E > 0.0) || (AtLo && E < 0.0);
        Deriv[Off + 0] = Freeze ? 0.0 : E;
        Deriv[Off + 1] = Nf * (E - Xd);
      }
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
      // Continuous LTI: dx = A·x + B·u, y = C·x (D = 0; the lowering
      // already marked D ≠ 0 as a non-loop-breaker we don't model yet).
      // SISO input; #345 added per-state x0 and a multi-row C → distinct
      // output ports out1..outP = (C·x)_1..(C·x)_P.
      const std::string *AS = paramS(B, "A");
      const std::string *BS = paramS(B, "B");
      const std::string *CS = paramS(B, "C");
      std::vector<double> A, Bm, Cm;
      int Ar = 0, Ac = 0, Br = 0, Bc = 0, Cr = 0, Cc = 0;
      if (AS) parseSimMatrix(*AS, A, Ar, Ac);
      if (BS) parseSimMatrix(*BS, Bm, Br, Bc);
      if (CS) parseSimMatrix(*CS, Cm, Cr, Cc);
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
      // y_r = (row r of C)·x. Each row drives output port out{r+1}; out1
      // also mirrors into the scalar Out_ for `out`-wired / scalar consumers.
      int Outs = Cr > 0 ? Cr : 1;
      for (int R = 0; R < Outs; ++R) {
        double Yr = 0.0;
        if (Cc > 0 && static_cast<int>(Cm.size()) >= (R + 1) * Cc)
          for (int Ki = 0; Ki < n && Ki < Cc; ++Ki)
            Yr += Cm[R * Cc + Ki] * State[Off + Ki];
        if (R == 0) Out_[I] = Yr;
        PortOut_[I]["out" + std::to_string(R + 1)] = Yr;
      }
    } else if (K == "signal_mpc_move") {
      /* MPC Toolbox Tier-3 §4.5 — MpcMove block.  Simulator carries
       * a static-gain MPC approximation: `u = gain · (r - ym)`.  The
       * full QP-solving form would link runtime_mpc.cpp into
       * MatlabFlowchart (Tier-3b carve-down — substantial dependency
       * chain expansion).  This block is sufficient to verify the
       * mflow infrastructure recognises MPC as a first-class block
       * kind, and to deploy an MPC-shaped lane-keeping demo through
       * the existing simulate / emit-c / cocotb SIL paths. */
      double Gain = paramD(B, "gain", 1.0);
      double R_def = paramD(B, "r_default", 0.0);
      double Ym = inputOf(I, "ym");
      double Rr = 0.0;
      /* Reference port may be omitted — fall back to r_default. */
      bool HasRef = false;
      for (auto &P : Inputs_[I])
        if (P.DstPort == "r") { HasRef = true; break; }
      if (HasRef) Rr = inputOf(I, "r");
      else Rr = R_def;
      Out_[I] = Gain * (Rr - Ym);
    } else if (K == "signal_world3d" || K == "signal_light3d" ||
               K == "signal_camera3d") {
      // mflow-3d-animation — scene config (world / light / camera). No ports,
      // no output; the emit lane reads their params directly. Nothing to eval.
      Out_[I] = 0.0;
    } else if (K == "signal_actor3d") {
      // mflow-3d-animation — gather the actor's transform into a width-9
      // sample [tx,ty,tz, rx,ry,rz (roll/pitch/yaw rad), sx,sy,sz]. Static
      // param defaults (translation/rotation/scale "x,y,z") are overridden
      // element-wise by any connected port; scale defaults to 1.
      double Tf[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
      parseVec3Param(B, "translation", &Tf[0]);
      parseVec3Param(B, "rotation", &Tf[3]);
      parseVec3Param(B, "scale", &Tf[6]);
      const char *Ports[3] = {"translation", "rotation", "scale"};
      for (int G = 0; G < 3; ++G) {
        if (!portConnected(I, Ports[G])) continue;
        for (int E = 0; E < 3; ++E) {
          double V = vecInput(I, Ports[G], E);
          if (!std::isnan(V)) Tf[G * 3 + E] = V;
        }
      }
      int W = OutWidth_[I] > 9 ? OutWidth_[I] : 9;
      VecOut_[I].assign(W, 0.0);
      for (int E = 0; E < 9; ++E) VecOut_[I][E] = Tf[E];
      // URDF actor (Tier 3b) — gather up to (W-9) joint angles from the
      // `jointAngles` vector port into the trailing slots.
      if (W > 9 && portConnected(I, "jointAngles")) {
        for (int Q = 0; Q < W - 9; ++Q) {
          double V = vecInput(I, "jointAngles", Q);
          VecOut_[I][9 + Q] = std::isnan(V) ? 0.0 : V;
        }
      }
      Out_[I] = Tf[0];
    } else if (K == "signal_scope3d") {
      // 3-D trajectory scope — gather the x/y/z input ports into a width-3
      // sample. The logging pass names the columns `<id>[x]/[y]/[z]` so a
      // 3-D path viewer can plot the trajectory.
      VecOut_[I].assign(3, 0.0);
      VecOut_[I][0] = inputOf(I, "x");
      VecOut_[I][1] = inputOf(I, "y");
      VecOut_[I][2] = inputOf(I, "z");
      Out_[I] = VecOut_[I][0];
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
        Out_[I] = Inputs_[I].empty() ? 0.0 : edgeValue(Inputs_[I].front());
      }
    } else if (K == "signal_demux" || K == "signal_switch") {
      // Algebra-only Tier-C stub: passthrough first input. Tier-E adds
      // the proper switch / demux semantics + zero-crossing.
      Out_[I] = Inputs_[I].empty() ? 0.0 : edgeValue(Inputs_[I].front());
    } else if (K == "signal_window" || K == "signal_fft" ||
               K == "signal_ifft" || K == "signal_spectrum" ||
               K == "signal_dwt" || K == "signal_idwt") {
      // #343 DSP frame transforms over a vector signal. Read `Want` elements
      // from the single input (a width-W source fills from VecOut_, a scalar
      // source broadcasts its value into element 0). Stateless: the whole
      // frame is recomputed from the current input each step.
      auto frame = [&](int Want) {
        std::vector<double> v(Want > 0 ? Want : 0, 0.0);
        if (Inputs_[I].empty() || Want <= 0) return v;
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1)
          for (int e = 0; e < Want && e < (int)VecOut_[Src].size(); ++e)
            v[e] = VecOut_[Src][e];
        else
          v[0] = Out_[Src];
        return v;
      };
      if (K == "signal_window") {
        int Nf = OutWidth_[I];
        std::vector<double> x = frame(Nf);
        const std::string *WT = paramS(B, "type");
        VecOut_[I].assign(Nf, 0.0);
        for (int k = 0; k < Nf; ++k)
          VecOut_[I][k] = windowCoef(WT, k, Nf) * x[k];
      } else if (K == "signal_fft") {
        // Real N-point DFT: X[k] = Σ_n x[n] e^{-j2πkn/N}, output packed as
        // [Re_0..Re_{N-1}, Im_0..Im_{N-1}] (width 2N). O(N²) — frames are
        // small; no FFTW dependency.
        int N2 = OutWidth_[I];
        int Nf = N2 / 2;
        std::vector<double> x = frame(Nf);
        VecOut_[I].assign(N2, 0.0);
        for (int kk = 0; kk < Nf; ++kk) {
          double re = 0.0, im = 0.0;
          for (int n = 0; n < Nf; ++n) {
            double ang = -2.0 * M_PI * kk * n / Nf;
            re += x[n] * std::cos(ang);
            im += x[n] * std::sin(ang);
          }
          VecOut_[I][kk] = re;
          VecOut_[I][Nf + kk] = im;
        }
      } else if (K == "signal_ifft") {
        // Inverse DFT of a complex [Re;Im] (width 2N) frame → real N-point
        // output: x[n] = (1/N) Σ_k (Re[k]cos − Im[k]sin)(2πkn/N).
        int Nf = OutWidth_[I];
        std::vector<double> X = frame(2 * Nf);
        VecOut_[I].assign(Nf, 0.0);
        for (int n = 0; n < Nf; ++n) {
          double re = 0.0;
          for (int kk = 0; kk < Nf; ++kk) {
            double ang = 2.0 * M_PI * kk * n / Nf;
            re += X[kk] * std::cos(ang) - X[Nf + kk] * std::sin(ang);
          }
          VecOut_[I][n] = Nf > 0 ? re / Nf : 0.0;
        }
      } else if (K == "signal_spectrum") {
        // Power spectrum of a real N-frame: |X[k]|² = Re[k]² + Im[k]² (width N).
        int Nf = OutWidth_[I];
        std::vector<double> x = frame(Nf);
        VecOut_[I].assign(Nf, 0.0);
        for (int kk = 0; kk < Nf; ++kk) {
          double re = 0.0, im = 0.0;
          for (int n = 0; n < Nf; ++n) {
            double ang = -2.0 * M_PI * kk * n / Nf;
            re += x[n] * std::cos(ang);
            im += x[n] * std::sin(ang);
          }
          VecOut_[I][kk] = re * re + im * im;
        }
      } else if (K == "signal_dwt") {
        // 1-level Haar DWT: approx[k] = (x[2k]+x[2k+1])/√2, detail[k] =
        // (x[2k]−x[2k+1])/√2. Output packs [approx (N/2); detail (N/2)].
        int Nf = OutWidth_[I];
        int H = Nf / 2;
        std::vector<double> x = frame(Nf);
        VecOut_[I].assign(Nf, 0.0);
        const double s = std::sqrt(2.0);
        for (int k = 0; k < H; ++k) {
          VecOut_[I][k] = (x[2 * k] + x[2 * k + 1]) / s;
          VecOut_[I][H + k] = (x[2 * k] - x[2 * k + 1]) / s;
        }
      } else { // signal_idwt
        // Inverse Haar: x[2k] = (a[k]+d[k])/√2, x[2k+1] = (a[k]−d[k])/√2,
        // from the [approx; detail] frame. signal_dwt → signal_idwt = identity.
        int Nf = OutWidth_[I];
        int H = Nf / 2;
        std::vector<double> X = frame(Nf);
        VecOut_[I].assign(Nf, 0.0);
        const double s = std::sqrt(2.0);
        for (int k = 0; k < H; ++k) {
          VecOut_[I][2 * k] = (X[k] + X[H + k]) / s;
          VecOut_[I][2 * k + 1] = (X[k] - X[H + k]) / s;
        }
      }
      Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
    } else if (K == "signal_color_space") {
      // #343 Vision — RGB↔grayscale over interleaved triples. rgb2gray:
      // out[i] = 0.299·R + 0.587·G + 0.114·B per pixel (width 3W→W).
      // gray2rgb: replicate each gray to R=G=B (width W→3W).
      std::vector<double> in;
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        in.assign(OutWidth_[Src], 0.0);
        if (OutWidth_[Src] > 1)
          for (int e = 0; e < OutWidth_[Src] && e < (int)VecOut_[Src].size(); ++e)
            in[e] = VecOut_[Src][e];
        else if (!in.empty())
          in[0] = Out_[Src];
      }
      const std::string *M = paramS(B, "mode");
      bool ToGray = !M || *M == "rgb2gray";
      int W = OutWidth_[I];
      VecOut_[I].assign(W, 0.0);
      if (ToGray) {
        for (int p = 0; p < W; ++p) {
          double r = (3 * p < (int)in.size()) ? in[3 * p] : 0.0;
          double g = (3 * p + 1 < (int)in.size()) ? in[3 * p + 1] : 0.0;
          double b = (3 * p + 2 < (int)in.size()) ? in[3 * p + 2] : 0.0;
          VecOut_[I][p] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
      } else { // gray2rgb
        for (int p = 0; 3 * p + 2 < W; ++p) {
          double v = (p < (int)in.size()) ? in[p] : 0.0;
          VecOut_[I][3 * p] = v;
          VecOut_[I][3 * p + 1] = v;
          VecOut_[I][3 * p + 2] = v;
        }
      }
      Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
    } else if (K == "signal_image_source" || K == "signal_image_filter" ||
               K == "signal_threshold") {
      // #343 Vision — grayscale image blocks over the flattened row-major 2-D
      // signal (width = rows·cols, shape in OutRows_/OutCols_). All stateless.
      if (K == "signal_image_source") {
        // Emit the constant image from `data` (any layout; flattened
        // row-major). Shape/width were stamped at lowering.
        int W = OutWidth_[I];
        VecOut_[I].assign(W, 0.0);
        if (const std::string *D = paramS(B, "data")) {
          std::vector<double> Vals; int r = 0, c = 0;
          parseSimMatrix(*D, Vals, r, c);
          for (int e = 0; e < W && e < (int)Vals.size(); ++e)
            VecOut_[I][e] = Vals[e];
        }
        Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
      } else {
        // filter / threshold: read the input image + its (rows, cols[, channels])
        // shape. A rank-3 input is an interleaved colour image — channels run
        // innermost: pixel (r,c,ch) lives at flat ((r*Cols + c)*Ch + ch).
        int Rows = 1, Cols = 1, Ch = 1;
        std::vector<double> Img;
        if (!Inputs_[I].empty()) {
          size_t Src = Inputs_[I].front().SrcBlock;
          const std::vector<int> &Sh = OutShape_[Src];
          if (Sh.size() >= 3) { Rows = Sh[0]; Cols = Sh[1]; Ch = Sh[2]; }
          else {
            Rows = OutRows_[Src] > 0 ? OutRows_[Src] : 1;
            Cols = OutCols_[Src] > 0 ? OutCols_[Src] : OutWidth_[Src];
          }
          Img.assign(OutWidth_[Src], 0.0);
          if (OutWidth_[Src] > 1)
            for (int e = 0; e < OutWidth_[Src] && e < (int)VecOut_[Src].size(); ++e)
              Img[e] = VecOut_[Src][e];
          else if (!Img.empty())
            Img[0] = Out_[Src];
        }
        int W = Rows * Cols * Ch;
        VecOut_[I].assign(W, 0.0);
        if (K == "signal_threshold") {
          // Element-wise over the whole (possibly multi-channel) buffer.
          double L = paramD(B, "level", 0.5);
          for (int e = 0; e < W && e < (int)Img.size(); ++e)
            VecOut_[I][e] = (Img[e] > L) ? 1.0 : 0.0;
        } else { // signal_image_filter — 2-D correlation, zero-padded borders,
                 // applied independently per channel for colour images.
          std::vector<double> Ker; int Kr = 0, Kc = 0;
          if (const std::string *KS = paramS(B, "kernel"))
            parseSimMatrix(*KS, Ker, Kr, Kc);
          if (Ker.empty()) {
            const std::string *T = paramS(B, "type");
            namedKernel(T ? *T : "box", Ker, Kr, Kc);
          }
          int ar = Kr / 2, ac = Kc / 2; // kernel anchor (center)
          for (int ch = 0; ch < Ch; ++ch)
            for (int r = 0; r < Rows; ++r)
              for (int c = 0; c < Cols; ++c) {
                double acc = 0.0;
                for (int i = 0; i < Kr; ++i)
                  for (int j = 0; j < Kc; ++j) {
                    int rr = r + i - ar, cc = c + j - ac;
                    if (rr < 0 || rr >= Rows || cc < 0 || cc >= Cols) continue;
                    acc += Img[(rr * Cols + cc) * Ch + ch] * Ker[i * Kc + j];
                  }
                VecOut_[I][(r * Cols + c) * Ch + ch] = acc;
              }
        }
        Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
      }
    } else if (K == "signal_reshape" || K == "signal_squeeze") {
      // §17.5 #9 / mflow-nd-signals — copy the upstream flat buffer
      // through; the shape change (reshape's target shape, or squeeze
      // dropping singleton dims) is a pure metadata switch — OutShape /
      // OutRows / OutCols were stamped by lowering, the row-major byte
      // layout is identical. The element-count contract is enforced
      // at lowering time.
      if (Inputs_[I].empty()) {
        if (OutWidth_[I] > 1)
          std::fill(VecOut_[I].begin(), VecOut_[I].end(), 0.0);
        Out_[I] = 0.0;
      } else {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[I] > 1) {
          VecOut_[I].assign(OutWidth_[I], 0.0);
          int SW = OutWidth_[Src];
          if (SW == 1) {
            // Scalar input broadcast — replicate across the new shape.
            std::fill(VecOut_[I].begin(), VecOut_[I].end(), Out_[Src]);
          } else {
            const auto &SV = VecOut_[Src];
            for (int E = 0; E < OutWidth_[I] && E < (int)SV.size(); ++E)
              VecOut_[I][E] = SV[E];
          }
          Out_[I] = VecOut_[I].front();
        } else {
          Out_[I] = Out_[Src];
        }
      }
    } else if (K == "signal_permute") {
      // mflow-nd-signals — reorder the axes of an N-D input per the 1-based
      // `order` permutation. Element movement: output axis k carries input
      // axis order[k]-1, so the output index o decodes (row-major over the
      // permuted OutShape) to an input multi-index whose axis order[k]-1
      // equals o's k-th coordinate. The order list was validated at lowering.
      if (Inputs_[I].empty()) {
        if (OutWidth_[I] > 1)
          std::fill(VecOut_[I].begin(), VecOut_[I].end(), 0.0);
        Out_[I] = 0.0;
      } else {
        size_t Src = Inputs_[I].front().SrcBlock;
        VecOut_[I].assign(OutWidth_[I], 0.0);
        if (OutWidth_[Src] <= 1) {
          // Scalar permute is the identity.
          std::fill(VecOut_[I].begin(), VecOut_[I].end(), Out_[Src]);
          Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
        } else {
          std::vector<int> InSh = OutShape_[Src];
          if (InSh.empty()) InSh = {OutWidth_[Src]};
          const std::vector<int> &OutSh = OutShape_[I];
          int N = (int)InSh.size();
          std::vector<int> Ord;
          if (const std::string *OP = paramS(B, "order")) {
            std::vector<double> Tmp; int r = 0, c = 0;
            parseSimMatrix(*OP, Tmp, r, c);
            for (double d : Tmp) Ord.push_back((int)(d + 0.5));
          }
          if ((int)Ord.size() != N) { Ord.clear();
            for (int k = 0; k < N; ++k) Ord.push_back(k + 1); }
          // Row-major strides for input and (permuted) output shapes.
          std::vector<int> InStr(N, 1);
          for (int d = N - 2; d >= 0; --d) InStr[d] = InStr[d + 1] * InSh[d + 1];
          int OutN = (int)OutSh.size();
          std::vector<int> OutStr(OutN > 0 ? OutN : 1, 1);
          for (int d = OutN - 2; d >= 0; --d)
            OutStr[d] = OutStr[d + 1] * OutSh[d + 1];
          const auto &SV = VecOut_[Src];
          int W = OutWidth_[I];
          for (int o = 0; o < W; ++o) {
            int Rem = o, InFlat = 0;
            for (int k = 0; k < OutN; ++k) {
              int Oi = Rem / OutStr[k];
              Rem %= OutStr[k];
              int InAxis = Ord[k] - 1; // output axis k came from input axis
              if (InAxis >= 0 && InAxis < N) InFlat += Oi * InStr[InAxis];
            }
            if (InFlat >= 0 && InFlat < (int)SV.size())
              VecOut_[I][o] = SV[InFlat];
          }
          Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
        }
      }
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
    } else if (K == "signal_awgn") {
      // #343 — Communications AWGN channel. y[n] = x[n] + n[n], with
      // n ~ N(0, σ²), σ² = signalPower / 10^(snr/10) (the Simulink AWGN
      // Channel "SNR + input signal power" mode). Reuses the per-block
      // xorshift64 + Box-Muller Gaussian generator (same seed state as
      // signal_noise). For a vector input (e.g. a complex [I, Q] symbol from a
      // modulator) each component gets an independent N(0, σ²) draw, so the
      // block models a noisy link end-to-end with PSK/QAM mod/demod.
      double Snr = paramD(B, "snr", 10.0);
      double Sp = paramD(B, "signalPower", 1.0);
      if (Sp < 0.0) Sp = 0.0;
      double Sigma = std::sqrt(Sp / std::pow(10.0, Snr / 10.0));
      uint64_t &S = NoiseSeed_[I];
      auto gauss = [&]() {
        S ^= S << 13; S ^= S >> 7; S ^= S << 17;
        double U1 = (S >> 11) / 9007199254740992.0;
        uint64_t S2 = S ^ 0xDEADBEEFCAFEBABEULL;
        S2 ^= S2 << 13; S2 ^= S2 >> 7; S2 ^= S2 << 17;
        double U2 = (S2 >> 11) / 9007199254740992.0;
        if (U1 < 1e-12) U1 = 1e-12;
        return std::sqrt(-2.0 * std::log(U1)) * std::cos(2.0 * M_PI * U2);
      };
      if (OutWidth_[I] > 1) {
        int W = OutWidth_[I];
        std::vector<double> X(W, 0.0);
        if (!Inputs_[I].empty()) {
          size_t Src = Inputs_[I].front().SrcBlock;
          if (OutWidth_[Src] > 1)
            for (int e = 0; e < W && e < (int)VecOut_[Src].size(); ++e)
              X[e] = VecOut_[Src][e];
          else
            X[0] = Out_[Src];
        }
        VecOut_[I].assign(W, 0.0);
        for (int e = 0; e < W; ++e) VecOut_[I][e] = X[e] + Sigma * gauss();
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = inputOf(I, "in") + Sigma * gauss();
      }
    } else if (K == "signal_error_rate") {
      // #343 — Communications error-rate (BER) sink. Output is the running
      // ratio of symbol mismatches between the `tx`/`rx` inputs. The compare +
      // accumulate happens once per major step in commitDigitalRegisters()
      // (evalAll runs multiple times per RK4 step and would over-count); here
      // we only surface the accumulated ratio so it logs and feeds downstream.
      Out_[I] = (TotAccum_[I] > 0.0) ? (ErrAccum_[I] / TotAccum_[I]) : 0.0;
    } else if (K == "signal_psk_mod" || K == "signal_qam_mod" ||
               K == "signal_psk_demod" || K == "signal_qam_demod") {
      // #343 Communications — PSK / QAM constellation map (modulator: symbol
      // index → complex [I, Q] vector, width 2) and demap (demodulator: I/Q
      // vector → nearest symbol index, hard decision). Stateless, both ways
      // are exact inverses on clean points, so mod → demod is the identity.
      int M = static_cast<int>(paramD(B, "M", 4.0));
      if (M < 2) M = 2;
      auto readIQ = [&](double &Iv, double &Qv) {
        Iv = 0.0; Qv = 0.0;
        if (Inputs_[I].empty()) return;
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1) {
          const auto &V = VecOut_[Src];
          if (!V.empty()) Iv = V[0];
          if (V.size() > 1) Qv = V[1];
        } else {
          Iv = Out_[Src];
        }
      };
      if (K == "signal_psk_mod") {
        // M-PSK: symbol k → exp(j(2πk/M + φ)), unit amplitude.
        double Phi = paramD(B, "phaseOffset", 0.0);
        int k = static_cast<int>(std::llround(inputOf(I, "in")));
        k = ((k % M) + M) % M;
        double Ang = 2.0 * M_PI * k / M + Phi;
        VecOut_[I].assign(2, 0.0);
        VecOut_[I][0] = std::cos(Ang);
        VecOut_[I][1] = std::sin(Ang);
        Out_[I] = VecOut_[I][0];
      } else if (K == "signal_psk_demod") {
        // Nearest-angle hard decision: k = round((∠(I,Q) − φ)·M/2π) mod M.
        double Phi = paramD(B, "phaseOffset", 0.0);
        double Iv, Qv; readIQ(Iv, Qv);
        double Ang = std::atan2(Qv, Iv) - Phi;
        int k = static_cast<int>(std::llround(Ang * M / (2.0 * M_PI)));
        Out_[I] = static_cast<double>(((k % M) + M) % M);
      } else {
        // Square M-QAM on the L×L grid (L = √M). Symbol k → (iIdx, qIdx) with
        // iIdx = k mod L, qIdx = k div L; levels are the odd integers
        // ±1, ±3, …, ±(L−1). `normalize` scales to unit average power.
        int L = static_cast<int>(std::lround(std::sqrt((double)M)));
        if (L < 1) L = 1;
        bool Norm = paramD(B, "normalize", 0.0) > 0.5;
        double Scale = std::sqrt(2.0 * (M - 1) / 3.0);
        if (Scale <= 0.0) Scale = 1.0;
        if (K == "signal_qam_mod") {
          int k = static_cast<int>(std::llround(inputOf(I, "in")));
          k = ((k % M) + M) % M;
          double Ip = 2.0 * (k % L) - (L - 1);
          double Qp = 2.0 * (k / L) - (L - 1);
          if (Norm) { Ip /= Scale; Qp /= Scale; }
          VecOut_[I].assign(2, 0.0);
          VecOut_[I][0] = Ip;
          VecOut_[I][1] = Qp;
          Out_[I] = Ip;
        } else { // signal_qam_demod
          double Iv, Qv; readIQ(Iv, Qv);
          if (Norm) { Iv *= Scale; Qv *= Scale; }
          auto level = [&](double v) {
            int idx = static_cast<int>(std::lround((v + (L - 1)) / 2.0));
            if (idx < 0) idx = 0;
            if (idx > L - 1) idx = L - 1;
            return idx;
          };
          Out_[I] = static_cast<double>(level(Qv) * L + level(Iv));
        }
      }
    } else if (K == "signal_running_stats") {
      // #343 — streaming statistics over the input. The Welford state is
      // updated once per major step in commitDigitalRegisters(); here we only
      // surface the requested statistic (params.stat: "mean" | "var" | "std",
      // default "mean"). Sample variance uses (n-1); n<2 ⇒ 0.
      const std::string *Stat = paramS(B, "stat");
      double Var = (RunCount_[I] > 1.0) ? (RunM2_[I] / (RunCount_[I] - 1.0))
                                        : 0.0;
      if (Stat && *Stat == "var")
        Out_[I] = Var;
      else if (Stat && *Stat == "std")
        Out_[I] = std::sqrt(Var);
      else
        Out_[I] = RunMean_[I];
    } else if (K == "signal_kalman") {
      // #343 — discrete Kalman filter. The predict+update recursion runs once
      // per major step in commitDigitalRegisters(); here we only publish the
      // current N-vector state estimate (Out_ mirrors element 0 for scalar
      // consumers, VecOut_ carries the full vector).
      const KalmanState &KS = Kalman_[I];
      if (OutWidth_[I] > 1) {
        VecOut_[I].assign(OutWidth_[I], 0.0);
        for (int j = 0; j < OutWidth_[I] && j < (int)KS.X.size(); ++j)
          VecOut_[I][j] = KS.X[j];
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = KS.X.empty() ? 0.0 : KS.X.front();
      }
    } else if (K == "signal_lqr") {
      // #343 Control — static state-feedback gain u = -K·x (LQR / pole
      // placement). K is an M×N matrix literal; the state x is the (vector)
      // input. Stateless, direct-feedthrough. A `sign` param of +1 emits +K·x.
      std::vector<double> Kmat;
      int Kr = 0, Kc = 0;
      if (const std::string *S = paramS(B, "K")) parseSimMatrix(*S, Kmat, Kr, Kc);
      double Sgn = paramD(B, "sign", -1.0) >= 0.0 ? 1.0 : -1.0;
      // Gather the N-element state input (vector source or scalar).
      std::vector<double> x(Kc > 0 ? Kc : 1, 0.0);
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1)
          for (int e = 0; e < (int)x.size() && e < (int)VecOut_[Src].size(); ++e)
            x[e] = VecOut_[Src][e];
        else
          x[0] = Out_[Src];
      }
      if (Kr > 0 && Kc > 0) {
        std::vector<double> u = matMul(Kmat, Kr, Kc, x, Kc, 1);
        VecOut_[I].assign(Kr, 0.0);
        for (int r = 0; r < Kr; ++r) VecOut_[I][r] = Sgn * u[r];
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = 0.0;
      }
    } else if (K == "signal_rf_2port") {
      // #343 RF — memoryless scattering 2-port: b = S·a, S a real 2×2 S-matrix
      // ([S11 S12; S21 S22]), a = [a1, a2] the incident-wave vector input.
      // Cascade instances to build an ideal time-domain network.
      std::vector<double> S;
      int Sr = 0, Sc = 0;
      if (const std::string *SS = paramS(B, "S")) parseSimMatrix(*SS, S, Sr, Sc);
      double a1 = 0.0, a2 = 0.0;
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1) {
          if (!VecOut_[Src].empty()) a1 = VecOut_[Src][0];
          if (VecOut_[Src].size() > 1) a2 = VecOut_[Src][1];
        } else {
          a1 = Out_[Src];
        }
      }
      VecOut_[I].assign(2, 0.0);
      if (Sr == 2 && Sc == 2) {
        VecOut_[I][0] = S[0] * a1 + S[1] * a2; // b1 = S11·a1 + S12·a2
        VecOut_[I][1] = S[2] * a1 + S[3] * a2; // b2 = S21·a1 + S22·a2
      }
      Out_[I] = VecOut_[I].front();
    } else if (K == "signal_pose_transform") {
      // #343 Nav/Robotics — apply a 2-D rigid-body pose to a point:
      //   out = R(theta)·[px, py] + [x, y]
      // The point [px, py] is the (vector) input; the pose (x, y, theta in rad)
      // is from params. Useful for body→world frame conversion in a fusion /
      // SLAM loop.
      double Px = 0.0, Py = 0.0;
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1) {
          if (!VecOut_[Src].empty()) Px = VecOut_[Src][0];
          if (VecOut_[Src].size() > 1) Py = VecOut_[Src][1];
        } else {
          Px = Out_[Src];
        }
      }
      double Th = paramD(B, "theta", 0.0);
      double Tx = paramD(B, "x", 0.0);
      double Ty = paramD(B, "y", 0.0);
      double c = std::cos(Th), s = std::sin(Th);
      VecOut_[I].assign(2, 0.0);
      VecOut_[I][0] = c * Px - s * Py + Tx;
      VecOut_[I][1] = s * Px + c * Py + Ty;
      Out_[I] = VecOut_[I].front();
    } else if (K == "signal_dnn_predict") {
      // #343 Deep Learning — one-hidden-layer MLP inference, in the loop:
      //   y = W2·act(W1·x + b1) + b2
      // W1 (H×N), b1 (H), W2 (M×H), b2 (M) are matrix literals; `activation`
      // is relu (default) / tanh / sigmoid / linear. Stateless feedthrough;
      // reuses the dense matMul kernel.
      std::vector<double> W1, B1, W2, B2;
      int w1r = 0, w1c = 0, b1r = 0, b1c = 0, w2r = 0, w2c = 0, b2r = 0, b2c = 0;
      if (const std::string *S = paramS(B, "W1")) parseSimMatrix(*S, W1, w1r, w1c);
      if (const std::string *S = paramS(B, "b1")) parseSimMatrix(*S, B1, b1r, b1c);
      if (const std::string *S = paramS(B, "W2")) parseSimMatrix(*S, W2, w2r, w2c);
      if (const std::string *S = paramS(B, "b2")) parseSimMatrix(*S, B2, b2r, b2c);
      const std::string *Act = paramS(B, "activation");
      auto activate = [&](double v) {
        const std::string a = Act ? *Act : "relu";
        if (a == "tanh") return std::tanh(v);
        if (a == "sigmoid") return 1.0 / (1.0 + std::exp(-v));
        if (a == "linear" || a == "none") return v;
        return v > 0.0 ? v : 0.0; // relu (default)
      };
      // Gather the N-element input.
      std::vector<double> x(w1c > 0 ? w1c : 1, 0.0);
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1)
          for (int e = 0; e < (int)x.size() && e < (int)VecOut_[Src].size(); ++e)
            x[e] = VecOut_[Src][e];
        else
          x[0] = Out_[Src];
      }
      if (w1r > 0 && w1c > 0 && w2r > 0 && w2c == w1r) {
        std::vector<double> h = matMul(W1, w1r, w1c, x, w1c, 1); // H×1
        for (int j = 0; j < w1r; ++j)
          h[j] = activate(h[j] + (j < (int)B1.size() ? B1[j] : 0.0));
        std::vector<double> y = matMul(W2, w2r, w2c, h, w2c, 1); // M×1
        VecOut_[I].assign(w2r, 0.0);
        for (int j = 0; j < w2r; ++j)
          VecOut_[I][j] = y[j] + (j < (int)B2.size() ? B2[j] : 0.0);
        Out_[I] = VecOut_[I].front();
      } else {
        Out_[I] = 0.0;
      }
    } else if (K == "signal_rl_agent") {
      // #343 Reinforcement Learning — a trained deterministic policy in the
      // loop. The same one-hidden-layer MLP as signal_dnn_predict maps the
      // state to a raw output; then `actionType` post-processes:
      //   discrete   → argmax index (a single scalar action, e.g. DQN)
      //   continuous → actionScale·tanh(raw) per output (bounded, e.g. DDPG)
      std::vector<double> W1, B1, W2, B2;
      int w1r = 0, w1c = 0, b1r = 0, b1c = 0, w2r = 0, w2c = 0, b2r = 0, b2c = 0;
      if (const std::string *S = paramS(B, "W1")) parseSimMatrix(*S, W1, w1r, w1c);
      if (const std::string *S = paramS(B, "b1")) parseSimMatrix(*S, B1, b1r, b1c);
      if (const std::string *S = paramS(B, "W2")) parseSimMatrix(*S, W2, w2r, w2c);
      if (const std::string *S = paramS(B, "b2")) parseSimMatrix(*S, B2, b2r, b2c);
      const std::string *Act = paramS(B, "activation");
      auto activate = [&](double v) {
        const std::string a = Act ? *Act : "relu";
        if (a == "tanh") return std::tanh(v);
        if (a == "sigmoid") return 1.0 / (1.0 + std::exp(-v));
        if (a == "linear" || a == "none") return v;
        return v > 0.0 ? v : 0.0;
      };
      std::vector<double> s(w1c > 0 ? w1c : 1, 0.0);
      if (!Inputs_[I].empty()) {
        size_t Src = Inputs_[I].front().SrcBlock;
        if (OutWidth_[Src] > 1)
          for (int e = 0; e < (int)s.size() && e < (int)VecOut_[Src].size(); ++e)
            s[e] = VecOut_[Src][e];
        else
          s[0] = Out_[Src];
      }
      std::vector<double> raw;
      if (w1r > 0 && w1c > 0 && w2r > 0 && w2c == w1r) {
        std::vector<double> h = matMul(W1, w1r, w1c, s, w1c, 1);
        for (int j = 0; j < w1r; ++j)
          h[j] = activate(h[j] + (j < (int)B1.size() ? B1[j] : 0.0));
        raw = matMul(W2, w2r, w2c, h, w2c, 1);
        for (int j = 0; j < w2r; ++j)
          raw[j] += (j < (int)B2.size() ? B2[j] : 0.0);
      }
      const std::string *AT = paramS(B, "actionType");
      bool Discrete = !AT || *AT == "discrete";
      if (raw.empty()) {
        Out_[I] = 0.0;
      } else if (Discrete) {
        int best = 0;
        for (int j = 1; j < (int)raw.size(); ++j)
          if (raw[j] > raw[best]) best = j;
        Out_[I] = static_cast<double>(best);
      } else {
        double Scale = paramD(B, "actionScale", 1.0);
        VecOut_[I].assign(raw.size(), 0.0);
        for (size_t j = 0; j < raw.size(); ++j)
          VecOut_[I][j] = Scale * std::tanh(raw[j]);
        Out_[I] = VecOut_[I].front();
      }
    } else if (K == "signal_dff" || K == "signal_tff" ||
               K == "signal_counter" || K == "signal_jkff" ||
               K == "signal_srff") {
      // #343 — clocked HDL registers. The output is the held value; the
      // edge-triggered state update happens once per major step in
      // commitDigitalRegisters() (NOT here — evalAll runs multiple times
      // per RK4 step, which would multiply-count a single clock edge).
      Out_[I] = DigitalLatch_[I];
    } else if (K == "signal_shift_register" || K == "signal_ram" ||
               K == "signal_rom") {
      // #343 HDL memory reads. shift_register → serial-out (last stage); RAM /
      // ROM → word at the `addr` input (clamped). The clocked shift / write
      // happens once per major step in commitDigitalRegisters().
      const auto &Mem = HdlMem_[I];
      if (Mem.empty()) {
        Out_[I] = 0.0;
      } else if (K == "signal_shift_register") {
        Out_[I] = Mem.back();
      } else {
        int Addr = (int)std::llround(inputOf(I, "addr"));
        if (Addr < 0) Addr = 0;
        if (Addr >= (int)Mem.size()) Addr = (int)Mem.size() - 1;
        Out_[I] = Mem[Addr];
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
    } else if (K == "signal_bus_creator") {
      // §17.5 #1 — pack N scalar inputs into a named-field vector
      // ordered by port id. Same wire shape as signal_mux but the
      // output carries an associated FieldNames map (set at
      // lowering time on B.FieldNames) so a downstream
      // signal_bus_selector can project by name.
      VecOut_[I].assign(OutWidth_[I], 0.0);
      std::vector<std::pair<int, size_t>> Order;
      for (size_t Idx = 0; Idx < Inputs_[I].size(); ++Idx) {
        const auto &P = Inputs_[I][Idx];
        int N = 1;
        if (P.DstPort.size() > 1 && P.DstPort[0] == 'i' &&
            P.DstPort[1] == 'n') {
          try { N = std::stoi(P.DstPort.substr(2)); }
          catch (...) { N = 1; }
        }
        Order.emplace_back(N, Idx);
      }
      std::sort(Order.begin(), Order.end());
      size_t Pos = 0;
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
      Out_[I] = VecOut_[I].empty() ? 0.0 : VecOut_[I].front();
    } else if (K == "signal_bus_selector") {
      // §17.5 #1 — look up `params.field` against the upstream
      // bus_creator's FieldNames map and project that element out.
      // Falls through to passthrough of the first element when no
      // matching field is found (matches Simulink's "default to
      // signal 1" fallback).
      const std::string *F = paramS(B, "field");
      size_t Src = static_cast<size_t>(-1);
      for (auto &P : Inputs_[I])
        if (P.DstPort == "in") { Src = P.SrcBlock; break; }
      double V = 0.0;
      if (Src != static_cast<size_t>(-1)) {
        const auto &SrcFields = M_.Blocks[Src].FieldNames;
        int Idx = 0;
        if (F && !F->empty()) {
          for (size_t K2 = 0; K2 < SrcFields.size(); ++K2) {
            if (SrcFields[K2] == *F) { Idx = (int)K2; break; }
          }
        }
        if (OutWidth_[Src] > 1 && Idx < (int)VecOut_[Src].size())
          V = VecOut_[Src][Idx];
        else
          V = Out_[Src];
      }
      Out_[I] = V;
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
    } else if (K == "signal_lookup_nd") {
      // N-D multilinear interpolation: per-axis clamp → (Lo, Frac), then blend
      // the 2^N table corners. Inputs in1..inN, table row-major (dim 0 outer).
      const auto &C = LookupNDCache_[I];
      if (!C.Valid) {
        Out_[I] = 0.0;
      } else {
        int N = static_cast<int>(C.Axes.size());
        std::vector<size_t> Lo(N, 0);
        std::vector<double> Frac(N, 0.0);
        for (int d = 0; d < N; ++d) {
          double X = inputOf(I, ("in" + std::to_string(d + 1)).c_str());
          const auto &Bp = C.Axes[d];
          if (Bp.size() < 2 || X <= Bp.front()) {
            Lo[d] = 0; Frac[d] = 0.0;
          } else if (X >= Bp.back()) {
            Lo[d] = Bp.size() - 2; Frac[d] = 1.0;
          } else {
            auto It = std::upper_bound(Bp.begin(), Bp.end(), X);
            Lo[d] = static_cast<size_t>(It - Bp.begin()) - 1;
            Frac[d] = (X - Bp[Lo[d]]) / (Bp[Lo[d] + 1] - Bp[Lo[d]]);
          }
        }
        std::vector<size_t> Stride(N, 1);
        for (int d = N - 2; d >= 0; --d)
          Stride[d] = Stride[d + 1] * C.Axes[d + 1].size();
        double Acc = 0.0;
        for (int Corner = 0; Corner < (1 << N); ++Corner) {
          double W = 1.0;
          size_t Idx = 0;
          bool Ok = true;
          for (int d = 0; d < N; ++d) {
            int Bit = (Corner >> d) & 1;
            W *= Bit ? Frac[d] : (1.0 - Frac[d]);
            size_t Ax = Lo[d] + Bit;
            if (Ax >= C.Axes[d].size()) { Ok = false; break; }
            Idx += Ax * Stride[d];
          }
          if (Ok && Idx < C.Z.size()) Acc += W * C.Z[Idx];
        }
        Out_[I] = Acc;
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
               K == "signal_biquad" ||
               K == "signal_lowpass" || K == "signal_highpass" ||
               K == "signal_dcblock" ||
               K == "signal_rate_transition") {
      // All read the same single-scalar latch as Unit Delay / ZOH (for the
      // filters it's the most-recent output y[n]). The fireDiscreteTicks
      // scheduler is what advances the state on each sample tick.
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
      // #354 — if this block has armed source-line breakpoints, route through
      // the AST interpreter (the JIT path has no per-statement hook) so the
      // body's lines can be watched and a hit recorded for the simulate-DAP.
      const std::set<int> *Watch = nullptr;
      {
        auto It = SourceBreakpoints_.find(M_.Blocks[I].Id);
        if (It != SourceBreakpoints_.end() && !It->second.empty())
          Watch = &It->second;
      }
      if (Watch && MatlabFnCache_[I]) {
        int Hit = -1, HitIdx = -1;
        std::map<std::string, double> HitVars;
        std::vector<double> Ys = runMatlabFunction(
            *MatlabFnCache_[I], U, T, Watch, &Hit, &HitVars, -1, &HitIdx);
        Out_[I] = Ys.empty() ? 0.0 : Ys[0];
        for (size_t K = 0; K < Ys.size(); ++K)
          PortOut_[I]["out" + std::to_string(K + 1)] = Ys[K];
        if (Hit >= 0 && LastSourceHit_.Line < 0) {
          LastSourceHit_.BlockId = M_.Blocks[I].Id;
          LastSourceHit_.Line = Hit;
          LastSourceHit_.Vars = std::move(HitVars);
          // #386 — capture the deterministic-replay context so the simulate-DAP
          // can step statement-by-statement through this body invocation.
          PendingStep_.BlockId = M_.Blocks[I].Id;
          PendingStep_.Inputs = U;
          PendingStep_.T = T;
          PendingStep_.Cache = MatlabFnCache_[I].get();
          PendingStep_.StmtIndex = HitIdx;
          PendingStep_.Valid = true;
        }
      } else if (MatlabFnJit_[I]) {
        // §17.5 #8 — JIT'd entrypoint. Pad inputs out to the arity
        // the wrapper expects (the unused trailing slots are zero,
        // matching the AST interpreter's behaviour for missing
        // inputs).
        double Pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        unsigned Arity = MatlabFnJitArity_[I];
        for (unsigned K = 0; K < Arity && K < U.size(); ++K) Pad[K] = U[K];
        Out_[I] = MatlabFcnJitOps_.Call(MatlabFnJit_[I], Pad, Arity);
      } else if (MatlabFnCache_[I]) {
        // #344: the AST interpreter returns every declared output; out1
        // mirrors into the scalar Out_, out2..outM publish on PortOut_.
        std::vector<double> Ys = runMatlabFunction(*MatlabFnCache_[I], U, T);
        Out_[I] = Ys.empty() ? 0.0 : Ys[0];
        for (size_t K = 0; K < Ys.size(); ++K)
          PortOut_[I]["out" + std::to_string(K + 1)] = Ys[K];
      } else {
        Out_[I] = evalMatlabFcn(MatlabFcnCache_[I].get(), U, T);
      }
    } else {
      // Loader-level reserved kinds are rejected at lowering, so
      // anything reaching here is an evaluator gap — treat as
      // passthrough so simulation doesn't crash and the user sees
      // the wrong-but-finite result instead of a segfault.
      Out_[I] = Inputs_[I].empty() ? 0.0 : edgeValue(Inputs_[I].front());
    }
  }
}

void MflowLinkSim::derivative(double T, const double *State, double *Deriv) {
  // mflow-variable-step-stiff-solvers — settle every block output from
  // `State` BEFORE reading derivatives. A continuous-state block computes
  // its derivative from its input (`Deriv[Off] = inputOf(...)`), which
  // reads the *current* `Out_` of the source block. With two mutually
  // coupled state blocks (e.g. a harmonic oscillator: x' = v, v' = -x), a
  // single pass would let whichever block is visited first read a stale
  // cross-coupled output left over from the previous evaluation. The
  // explicit RK lanes tolerate that one-evaluation lag, but the implicit
  // Jacobian divides the O(state) staleness by the FD step (~1e-7) and
  // explodes. A settling pass (Deriv = null) makes every `Out_` current,
  // so the derivative pass reads a coherent state.
  if (Deriv) evalAll(T, State, nullptr);
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
  const double Tol = std::max(EffectiveRelTol_, 1e-8);
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
// The default for `fixed_step` models and the last-resort fallback when an
// adaptive step fails to converge. The variable-step lane uses the embedded
// pairs below (Dormand-Prince 5(4) for `ode45`, Bogacki-Shampine 3(2) for
// `ode23`) and the implicit BDF1 lane handles `ode15s`. The user-facing
// `solver.algorithm` selects among them and surfaces in `--dry-run` output.
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

//===-----------------------------------------------------------------===//
// mflow-variable-step-stiff-solvers — Bogacki-Shampine 3(2) adaptive
// integrator (the standard `ode23` tableau). Returns the 3rd-order
// solution in `YOut` and the embedded 2nd-order error in `Err`. FSAL:
// stage 4 is evaluated at `YOut` (and would seed the next step's k1).
// See Bogacki & Shampine (1989); the same pair MATLAB's `ode23` uses.
//===-----------------------------------------------------------------===//

struct BS32Workspace {
  std::vector<double> K1, K2, K3, K4, Yt;
};

void bs32Step(MflowLinkSim &Sim,
              void (MflowLinkSim::*Deriv)(double, const double *, double *),
              double TBegin, double H,
              const std::vector<double> &YIn,
              std::vector<double> &YOut,
              std::vector<double> &Err,
              BS32Workspace &W) {
  const size_t Nx = YIn.size();
  if (W.K1.size() != Nx) {
    W.K1.assign(Nx, 0.0); W.K2.assign(Nx, 0.0);
    W.K3.assign(Nx, 0.0); W.K4.assign(Nx, 0.0); W.Yt.assign(Nx, 0.0);
  }
  // Stage 1: k1 = f(t, y).
  (Sim.*Deriv)(TBegin, YIn.data(), W.K1.data());
  // Stage 2: t + h/2, y + h·(1/2)·k1.
  for (size_t I = 0; I < Nx; ++I) W.Yt[I] = YIn[I] + H * 0.5 * W.K1[I];
  (Sim.*Deriv)(TBegin + 0.5 * H, W.Yt.data(), W.K2.data());
  // Stage 3: t + 3h/4, y + h·(3/4)·k2.
  for (size_t I = 0; I < Nx; ++I) W.Yt[I] = YIn[I] + H * 0.75 * W.K2[I];
  (Sim.*Deriv)(TBegin + 0.75 * H, W.Yt.data(), W.K3.data());
  // 3rd-order solution: y_{n+1} = y + h·(2/9 k1 + 1/3 k2 + 4/9 k3).
  for (size_t I = 0; I < Nx; ++I)
    YOut[I] = YIn[I] + H * ((2.0 / 9.0) * W.K1[I]
                            + (1.0 / 3.0) * W.K2[I]
                            + (4.0 / 9.0) * W.K3[I]);
  // Stage 4 (FSAL): k4 = f(t + h, y_{n+1}).
  (Sim.*Deriv)(TBegin + H, YOut.data(), W.K4.data());
  // Embedded 2nd-order estimate via the coefficient differences
  // (3rd-order b − 2nd-order b̂): −5/72 k1 + 1/12 k2 + 1/9 k3 − 1/8 k4.
  Err.assign(Nx, 0.0);
  for (size_t I = 0; I < Nx; ++I)
    Err[I] = H * ((-5.0 / 72.0) * W.K1[I]
                  + (1.0 / 12.0) * W.K2[I]
                  + (1.0 / 9.0)  * W.K3[I]
                  - (1.0 / 8.0)  * W.K4[I]);
}

//===-----------------------------------------------------------------===//
// §17.5 #3 — Implicit Backward Euler (BDF1) substep for stiff
// systems. Solves `y_new = y_old + h · f(t_old + h, y_new)` via a
// Newton iteration with a forward-difference Jacobian and a dense LU
// solve, starting from an explicit-Euler predictor.
//
// L-stable: the test equation `y' = λ·y` with `Re(λ) → -∞` gives
// `y_new = y_old / (1 - h·λ)` → 0 — the right behaviour for stiff
// dissipation. DOPRI5 needs `|h·λ| < 2.78` to stay stable; BDF1
// has no stability bound, so the user can pick a step size matched
// to accuracy rather than stability.
//
// This is a true Newton iteration (not fixed-point): each step forms
// `J = I − h·∂f/∂y` by forward-difference and solves `J·Δ = G` via the
// dense LU below. Current caveats (scoped in OpenSpec change
// mflow-variable-step-stiff-solvers): order 1 only, fixed step, and the
// Jacobian is recomputed every Newton iteration rather than amortised.
//===-----------------------------------------------------------===//

// Tiny dense LU + back-substitute for the Newton update. Returns
// false on singular/zero pivots — caller falls back to a smaller
// step.
bool solveDense(std::vector<double> &J, int N, std::vector<double> &B) {
  // J stored row-major: J[i*N+j]. Solve J·x = B in place; B receives x.
  for (int K = 0; K < N; ++K) {
    // Partial pivoting.
    int Pivot = K;
    double Best = std::fabs(J[K * N + K]);
    for (int I = K + 1; I < N; ++I) {
      double V = std::fabs(J[I * N + K]);
      if (V > Best) { Best = V; Pivot = I; }
    }
    if (Best < 1e-15) return false;
    if (Pivot != K) {
      for (int J2 = 0; J2 < N; ++J2)
        std::swap(J[K * N + J2], J[Pivot * N + J2]);
      std::swap(B[K], B[Pivot]);
    }
    for (int I = K + 1; I < N; ++I) {
      double F = J[I * N + K] / J[K * N + K];
      for (int J2 = K; J2 < N; ++J2)
        J[I * N + J2] -= F * J[K * N + J2];
      B[I] -= F * B[K];
    }
  }
  // Back-substitute.
  for (int I = N - 1; I >= 0; --I) {
    double S = B[I];
    for (int J2 = I + 1; J2 < N; ++J2) S -= J[I * N + J2] * B[J2];
    B[I] = S / J[I * N + I];
  }
  return true;
}

void bdf1Step(MflowLinkSim &Sim,
              void (MflowLinkSim::*Deriv)(double, const double *, double *),
              double TBegin, double H,
              const std::vector<double> &YIn,
              std::vector<double> &YOut,
              double RelTol, double AbsTol) {
  // Newton iteration on G(y) = y - y_old - h·f(t+h, y). The
  // Jacobian J = ∂G/∂y = I - h · ∂f/∂y is approximated by
  // forward-difference perturbation. For stiff y' = λ·y with
  // Re(λ) → -∞, J = 1 - h·λ → ∞ and the Newton update y_new =
  // y_old/(1 - h·λ) → 0 — exact L-stability.
  const int Nx = static_cast<int>(YIn.size());
  YOut.assign(Nx, 0.0);
  if (Nx == 0) return;
  std::vector<double> F(Nx), FP(Nx), G(Nx), Delta(Nx);
  // Predictor: explicit Euler.
  (Sim.*Deriv)(TBegin, YIn.data(), F.data());
  for (int I = 0; I < Nx; ++I) YOut[I] = YIn[I] + H * F[I];

  for (int It = 0; It < 20; ++It) {
    // Residual G(YOut).
    (Sim.*Deriv)(TBegin + H, YOut.data(), F.data());
    for (int I = 0; I < Nx; ++I) G[I] = YOut[I] - YIn[I] - H * F[I];

    // Convergence on G.
    double GNorm = 0.0;
    for (int I = 0; I < Nx; ++I) {
      double Sc = AbsTol + RelTol * std::fabs(YOut[I]);
      if (Sc < 1e-15) Sc = 1e-15;
      double E = std::fabs(G[I]) / Sc;
      if (E > GNorm) GNorm = E;
    }
    if (GNorm < 1.0) break;

    // Finite-difference Jacobian. Column j: J[:,j] = (G(y+eps·e_j)
    // − G(y)) / eps = e_j − h·(f(y+eps·e_j) − f(y))/eps. Reuses F
    // for the unperturbed f and FP for the perturbed.
    std::vector<double> J(Nx * Nx, 0.0);
    std::vector<double> YPert = YOut;
    for (int Jc = 0; Jc < Nx; ++Jc) {
      double Eps = 1e-7 * std::max(1.0, std::fabs(YOut[Jc]));
      YPert[Jc] += Eps;
      (Sim.*Deriv)(TBegin + H, YPert.data(), FP.data());
      YPert[Jc] = YOut[Jc];
      for (int I = 0; I < Nx; ++I) {
        double Dfdy = (FP[I] - F[I]) / Eps;
        J[I * Nx + Jc] = (I == Jc ? 1.0 : 0.0) - H * Dfdy;
      }
    }

    // Newton update: solve J·Δ = G, then YOut −= Δ.
    Delta = G;
    if (!solveDense(J, Nx, Delta)) break; // singular — bail
    for (int I = 0; I < Nx; ++I) YOut[I] -= Delta[I];
  }
}

//===-----------------------------------------------------------------===//
// mflow-variable-step-stiff-solvers — Trapezoidal rule (`ode23t`).
// Implicit, A-stable, 2nd order, and *non-dissipative*: unlike BDF1 it
// does not artificially damp oscillations, so it suits moderately-stiff
// oscillatory systems. Newton on the residual
//   G(y) = y − y_old − (h/2)·(f0 + f(t+h, y)),  J = I − (h/2)·∂f/∂y,
// reusing the forward-difference Jacobian + dense LU from `bdf1Step`.
// Fixed-step in this slice (user picks `maxStep`).
//===-----------------------------------------------------------------===//

void trapezoidalStep(MflowLinkSim &Sim,
                     void (MflowLinkSim::*Deriv)(double, const double *,
                                                 double *),
                     double TBegin, double H,
                     const std::vector<double> &YIn,
                     std::vector<double> &YOut,
                     double RelTol, double AbsTol) {
  const int Nx = static_cast<int>(YIn.size());
  YOut.assign(Nx, 0.0);
  if (Nx == 0) return;
  std::vector<double> F0(Nx), F(Nx), FP(Nx), G(Nx), Delta(Nx);
  // f0 = f(t0, y0) is fixed across the iteration (the explicit half).
  (Sim.*Deriv)(TBegin, YIn.data(), F0.data());
  // Predictor: explicit Euler.
  for (int I = 0; I < Nx; ++I) YOut[I] = YIn[I] + H * F0[I];

  for (int It = 0; It < 20; ++It) {
    (Sim.*Deriv)(TBegin + H, YOut.data(), F.data());
    for (int I = 0; I < Nx; ++I)
      G[I] = YOut[I] - YIn[I] - 0.5 * H * (F0[I] + F[I]);

    double GNorm = 0.0;
    for (int I = 0; I < Nx; ++I) {
      double Sc = AbsTol + RelTol * std::fabs(YOut[I]);
      if (Sc < 1e-15) Sc = 1e-15;
      double E = std::fabs(G[I]) / Sc;
      if (E > GNorm) GNorm = E;
    }
    if (GNorm < 1.0) break;

    // J = I − (h/2)·∂f/∂y by forward difference.
    std::vector<double> J(Nx * Nx, 0.0);
    std::vector<double> YPert = YOut;
    for (int Jc = 0; Jc < Nx; ++Jc) {
      double Eps = 1e-7 * std::max(1.0, std::fabs(YOut[Jc]));
      YPert[Jc] += Eps;
      (Sim.*Deriv)(TBegin + H, YPert.data(), FP.data());
      YPert[Jc] = YOut[Jc];
      for (int I = 0; I < Nx; ++I) {
        double Dfdy = (FP[I] - F[I]) / Eps;
        J[I * Nx + Jc] = (I == Jc ? 1.0 : 0.0) - 0.5 * H * Dfdy;
      }
    }

    Delta = G;
    if (!solveDense(J, Nx, Delta)) break;
    for (int I = 0; I < Nx; ++I) YOut[I] -= Delta[I];
  }
}

//===-----------------------------------------------------------------===//
// mflow-variable-step-stiff-solvers — modified Rosenbrock (2)3 stiff
// integrator: MATLAB's `ode23s` (Bank/Shampine-Reichelt). A one-step,
// linearly-implicit method — one Jacobian factorisation per step and no
// Newton loop (three back-substitutions against the same `W = I − h·d·J`).
// L-stable, 2nd order. `d = 1/(2+√2)`, `e32 = 6+√2`. The time-derivative
// term `T0 = ∂f/∂t` is finite-differenced (mflowLink models are usually
// non-autonomous: time-varying sources). Fixed-step in this slice (the
// user picks `maxStep`), matching the BDF1 lane; `Err` is filled for the
// variable-step controller that lands with the BDF slice.
//===-----------------------------------------------------------------===//

void rosenbrockStep(MflowLinkSim &Sim,
                    void (MflowLinkSim::*Deriv)(double, const double *,
                                                double *),
                    double TBegin, double H,
                    const std::vector<double> &YIn,
                    std::vector<double> &YOut,
                    std::vector<double> &Err) {
  const int Nx = static_cast<int>(YIn.size());
  YOut.assign(Nx, 0.0);
  Err.assign(Nx, 0.0);
  if (Nx == 0) return;
  const double D = 1.0 / (2.0 + std::sqrt(2.0));
  const double E32 = 6.0 + std::sqrt(2.0);

  std::vector<double> F0(Nx), F1(Nx), F2(Nx), FP(Nx), Yt(Nx);
  (Sim.*Deriv)(TBegin, YIn.data(), F0.data());

  // Time derivative T0 = ∂f/∂t by forward difference.
  std::vector<double> T0(Nx, 0.0);
  double DtEps = 1e-7 * std::max(1.0, std::fabs(TBegin));
  if (H > 0) {
    (Sim.*Deriv)(TBegin + DtEps, YIn.data(), FP.data());
    for (int I = 0; I < Nx; ++I) T0[I] = (FP[I] - F0[I]) / DtEps;
  }

  // W = I − h·d·J, J = ∂f/∂y by forward difference (column by column).
  std::vector<double> W(Nx * Nx, 0.0);
  std::vector<double> YPert = YIn;
  for (int Jc = 0; Jc < Nx; ++Jc) {
    double Eps = 1e-7 * std::max(1.0, std::fabs(YIn[Jc]));
    YPert[Jc] += Eps;
    (Sim.*Deriv)(TBegin, YPert.data(), FP.data());
    YPert[Jc] = YIn[Jc];
    for (int I = 0; I < Nx; ++I) {
      double Dfdy = (FP[I] - F0[I]) / Eps;
      W[I * Nx + Jc] = (I == Jc ? 1.0 : 0.0) - H * D * Dfdy;
    }
  }

  auto solveW = [&](std::vector<double> &Rhs) -> bool {
    std::vector<double> Wc = W; // solveDense factors in place
    return solveDense(Wc, Nx, Rhs);
  };

  // Stage 1: W·k1 = F0 + h·d·T0.
  std::vector<double> K1(Nx), K2(Nx), K3(Nx);
  for (int I = 0; I < Nx; ++I) K1[I] = F0[I] + H * D * T0[I];
  if (!solveW(K1)) { for (int I = 0; I < Nx; ++I) YOut[I] = YIn[I] + H * F0[I]; return; }

  // Stage 2: F1 = f(t + h/2, y + h/2·k1); W·(k2 − k1) = F1 − k1.
  for (int I = 0; I < Nx; ++I) Yt[I] = YIn[I] + 0.5 * H * K1[I];
  (Sim.*Deriv)(TBegin + 0.5 * H, Yt.data(), F1.data());
  std::vector<double> Dk(Nx);
  for (int I = 0; I < Nx; ++I) Dk[I] = F1[I] - K1[I];
  if (!solveW(Dk)) { for (int I = 0; I < Nx; ++I) YOut[I] = YIn[I] + H * K1[I]; return; }
  for (int I = 0; I < Nx; ++I) K2[I] = K1[I] + Dk[I];

  // 2nd-order solution: y1 = y0 + h·k2.
  for (int I = 0; I < Nx; ++I) YOut[I] = YIn[I] + H * K2[I];

  // Stage 3 (error): F2 = f(t+h, y1);
  // W·k3 = F2 − e32·(k2 − F1) − 2·(k1 − F0) + h·d·T0.
  (Sim.*Deriv)(TBegin + H, YOut.data(), F2.data());
  for (int I = 0; I < Nx; ++I)
    K3[I] = F2[I] - E32 * (K2[I] - F1[I]) - 2.0 * (K1[I] - F0[I]) + H * D * T0[I];
  if (!solveW(K3)) return; // keep y1, no error estimate
  // Embedded error: err = (h/6)·(k1 − 2·k2 + k3).
  for (int I = 0; I < Nx; ++I)
    Err[I] = (H / 6.0) * (K1[I] - 2.0 * K2[I] + K3[I]);
}

//===-----------------------------------------------------------------===//
// mflow-variable-step-stiff-solvers — TR-BDF2 (`ode23tb`). A two-stage
// composite: a trapezoidal sub-step to t+γ·h (γ = 2−√2), then a BDF2
// sub-step over the three points (t, y), (t+γh, y_γ), (t+h, y₁) to t+h.
// L-stable and 2nd order; the trapezoidal stage keeps it accurate while
// the BDF2 stage damps high-frequency stiff modes. Both stages reuse the
// Newton + forward-difference Jacobian + dense LU machinery; the BDF2
// coefficients come from the Lagrange derivative on the non-uniform mesh
// (α₂·y₁ + α₁·y_γ + α₀·y₀ − h·f = 0, with α₀+α₁+α₂ = 0 for consistency).
// Fixed-step in this slice.
//===-----------------------------------------------------------------===//

void trBDF2Step(MflowLinkSim &Sim,
                void (MflowLinkSim::*Deriv)(double, const double *, double *),
                double TBegin, double H,
                const std::vector<double> &YIn,
                std::vector<double> &YOut,
                double RelTol, double AbsTol) {
  const int Nx = static_cast<int>(YIn.size());
  YOut.assign(Nx, 0.0);
  if (Nx == 0) return;
  const double Gamma = 2.0 - std::sqrt(2.0);
  // Stage 1 (trapezoidal): a step of size γ·h → y_γ.
  std::vector<double> YG;
  trapezoidalStep(Sim, Deriv, TBegin, Gamma * H, YIn, YG, RelTol, AbsTol);

  // Stage 2 (BDF2): α₂·y₁ + α₁·y_γ + α₀·y₀ − h·f(t+h, y₁) = 0.
  const double A2 = (2.0 - Gamma) / (1.0 - Gamma);
  const double A1 = -1.0 / (Gamma * (1.0 - Gamma));
  const double A0 = (1.0 - Gamma) / Gamma;
  std::vector<double> F(Nx), FP(Nx), G(Nx), Delta(Nx);
  YOut = YG; // predictor

  for (int It = 0; It < 20; ++It) {
    (Sim.*Deriv)(TBegin + H, YOut.data(), F.data());
    for (int I = 0; I < Nx; ++I)
      G[I] = A2 * YOut[I] + A1 * YG[I] + A0 * YIn[I] - H * F[I];

    double GNorm = 0.0;
    for (int I = 0; I < Nx; ++I) {
      double Sc = AbsTol + RelTol * std::fabs(YOut[I]);
      if (Sc < 1e-15) Sc = 1e-15;
      double E = std::fabs(G[I]) / Sc;
      if (E > GNorm) GNorm = E;
    }
    if (GNorm < 1.0) break;

    std::vector<double> J(Nx * Nx, 0.0);
    std::vector<double> YPert = YOut;
    for (int Jc = 0; Jc < Nx; ++Jc) {
      double Eps = 1e-7 * std::max(1.0, std::fabs(YOut[Jc]));
      YPert[Jc] += Eps;
      (Sim.*Deriv)(TBegin + H, YPert.data(), FP.data());
      YPert[Jc] = YOut[Jc];
      for (int I = 0; I < Nx; ++I) {
        double Dfdy = (FP[I] - F[I]) / Eps;
        J[I * Nx + Jc] = (I == Jc ? A2 : 0.0) - H * Dfdy;
      }
    }
    Delta = G;
    if (!solveDense(J, Nx, Delta)) break;
    for (int I = 0; I < Nx; ++I) YOut[I] -= Delta[I];
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
    if (ImplicitAdaptive_) {
      // mflow-variable-step-stiff-solvers — adaptive BDF1 (`ode15s` under
      // variable_step). Sub-step the major window with step-size control
      // driven by a step-doubling (Richardson) error estimate: a full step
      // of `HTry` vs two half steps. For the order-1 method the difference
      // estimates the local error; advance with the (more accurate) half-step
      // result. The exponent is 1/(order+1) = 1/2. `maxStep` caps the step.
      const double RelTol = EffectiveRelTol_;
      const double AbsTol = EffectiveAbsTol_;
      double TLocal = T_;
      double TEnd   = T_ + H;
      double HCur   = std::min(CurrentAdaptiveH_, H);
      std::vector<double> YFull(Nx), YHalf1(Nx), YHalf2(Nx);
      const int MaxRetries = 32;
      while (TLocal < TEnd - 1e-15) {
        double HTry = std::min(HCur, TEnd - TLocal);
        if (HTry <= 1e-15) break;
        int Retry = 0;
        bool Accepted = false;
        while (!Accepted && Retry++ < MaxRetries) {
          bdf1Step(*this, &MflowLinkSim::derivative, TLocal, HTry, Y_, YFull,
                   RelTol, AbsTol);
          bdf1Step(*this, &MflowLinkSim::derivative, TLocal, 0.5 * HTry, Y_,
                   YHalf1, RelTol, AbsTol);
          bdf1Step(*this, &MflowLinkSim::derivative, TLocal + 0.5 * HTry,
                   0.5 * HTry, YHalf1, YHalf2, RelTol, AbsTol);
          double Norm = 0.0;
          for (size_t I = 0; I < Nx; ++I) {
            double Sc = AbsTol + RelTol *
                        std::max(std::fabs(Y_[I]), std::fabs(YHalf2[I]));
            if (Sc <= 0.0) Sc = AbsTol > 0 ? AbsTol : 1e-12;
            double E = std::fabs(YHalf2[I] - YFull[I]) / Sc;
            if (E > Norm) Norm = E;
          }
          if (Norm <= 1.0) {
            Accepted = true;
            Y1 = YHalf2; // advance with the more accurate half-step solution
            double Factor = (Norm > 1e-12) ? 0.9 * std::pow(Norm, -0.5) : 5.0;
            if (Factor > 5.0) Factor = 5.0;
            HCur = std::min(HTry * Factor, StepSize_);
          } else {
            double Factor = 0.9 * std::pow(Norm, -0.5);
            if (Factor < 0.1) Factor = 0.1;
            HTry = HTry * Factor;
            if (HTry < 1e-15) break;
          }
        }
        if (!Accepted) {
          // Last resort — take the plain Backward-Euler step and move on.
          bdf1Step(*this, &MflowLinkSim::derivative, TLocal, HTry, Y_, YFull,
                   RelTol, AbsTol);
          Y1 = YFull;
        }
        Y_ = Y1;
        TLocal += HTry;
      }
      CurrentAdaptiveH_ = HCur;
    } else if (Implicit_) {
      // §17.5 #3 / mflow-variable-step-stiff-solvers — fixed-step implicit
      // stiff lane. `ode15s` (BDF1, Backward Euler via Newton), `ode23s`
      // (modified Rosenbrock), `ode23t` (trapezoidal), `ode23tb` (TR-BDF2);
      // the user picks the step via `settings.solver.maxStep`.
      if (StiffMethod_ == StiffMethod::ROSENBROCK) {
        std::vector<double> Err(Nx, 0.0);
        rosenbrockStep(*this, &MflowLinkSim::derivative, T_, H, Y_, Y1, Err);
      } else if (StiffMethod_ == StiffMethod::TRAPEZOIDAL) {
        trapezoidalStep(*this, &MflowLinkSim::derivative, T_, H, Y_, Y1,
                        EffectiveRelTol_, EffectiveAbsTol_);
      } else if (StiffMethod_ == StiffMethod::TRBDF2) {
        trBDF2Step(*this, &MflowLinkSim::derivative, T_, H, Y_, Y1,
                   EffectiveRelTol_, EffectiveAbsTol_);
      } else {
        bdf1Step(*this, &MflowLinkSim::derivative, T_, H, Y_, Y1,
                 EffectiveRelTol_, EffectiveAbsTol_);
      }
      Y_ = std::move(Y1);
    } else if (AdaptiveSolver_) {
      // Item-2 / mflow-variable-step-stiff-solvers — embedded adaptive
      // step control. `ode45` runs Dormand-Prince 5(4); `ode23` runs the
      // native Bogacki-Shampine 3(2). Try the full window `H`; if the
      // embedded error estimate exceeds tolerance, shrink and retry. Each
      // accepted step updates `CurrentAdaptiveH_` so the next major step
      // starts from a reasonable size. The maxStep cap (`StepSize_`) is
      // respected: the chosen step never exceeds it.
      static thread_local DOPRI5Workspace WS;
      static thread_local BS32Workspace WS23;
      std::vector<double> Err(Nx, 0.0);
      double TLocal = T_;
      double TEnd   = T_ + H;
      double HCur   = std::min(CurrentAdaptiveH_, H);
      const double RelTol = EffectiveRelTol_;
      const double AbsTol = EffectiveAbsTol_;
      // Step-size exponent keyed off the embedded estimate's order:
      // 1/(order+1). DOPRI5 → 0.2, BS32 → 1/3.
      const double Expo = 1.0 / (AdaptiveErrOrder_ + 1.0);
      const int MaxRetries = 32;
      while (TLocal < TEnd - 1e-15) {
        double HTry = std::min(HCur, TEnd - TLocal);
        if (HTry <= 1e-15) break;
        int Retry = 0;
        bool Accepted = false;
        while (!Accepted && Retry++ < MaxRetries) {
          if (AdaptiveMethod_ == AdaptiveMethod::BS32)
            bs32Step(*this, &MflowLinkSim::derivative,
                     TLocal, HTry, Y_, Y1, Err, WS23);
          else
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
            // Grow step on success, capped at the configured maxStep.
            double Factor = (Norm > 1e-12)
                              ? 0.9 * std::pow(Norm, -Expo)
                              : 5.0;
            if (Factor > 5.0) Factor = 5.0;
            HCur = std::min(HTry * Factor, StepSize_);
          } else {
            // Reject + shrink. Lower bound keeps us from
            // infinite-looping on a hard discontinuity.
            double Factor = 0.9 * std::pow(Norm, -Expo);
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
  // §17.5 #2 — signal_integrator reset port. Scan for any
  // integrator with a connected `reset` port; if the reset source
  // just rose through zero (PrevOut ≤ 0 && Out > 0), reload the
  // continuous state from the `init` port (if connected) or
  // `params.initialCondition`. The reload happens BEFORE the
  // snapshot of PrevOut_ below, so the next step's edge detector
  // sees the just-completed step's outputs as "previous".
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    if (M_.Blocks[I].Kind != "signal_integrator") continue;
    size_t ResetSrc = static_cast<size_t>(-1);
    size_t InitSrc  = static_cast<size_t>(-1);
    for (auto &P : Inputs_[I]) {
      if (P.DstPort == "reset")     ResetSrc = P.SrcBlock;
      else if (P.DstPort == "init") InitSrc  = P.SrcBlock;
    }
    if (ResetSrc == static_cast<size_t>(-1)) continue;
    bool Rising = PrevOut_[ResetSrc] <= 0.0 && Out_[ResetSrc] > 0.0;
    if (!Rising) continue;
    double NewIC =
        (InitSrc != static_cast<size_t>(-1))
            ? Out_[InitSrc]
            : paramD(M_.Blocks[I], "initialCondition", 0.0);
    Y_[StateOffset_[I]] = NewIC;
  }
  // #343 — clocked HDL registers (signal_dff / signal_tff / signal_counter).
  // Done once per major step (here, not in evalAll) so a single clock edge
  // updates the register exactly once. Active-high `reset`/`rst` reloads the
  // initial value; otherwise a `clk` rising edge (PrevOut_ ≤ 0 && Out_ > 0)
  // samples/toggles/increments. The refresh evalAll below makes the new value
  // visible in this step's logged output (a posedge-triggered register).
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const std::string &K = M_.Blocks[I].Kind;
    if (K != "signal_dff" && K != "signal_tff" && K != "signal_counter" &&
        K != "signal_jkff" && K != "signal_srff")
      continue;
    auto srcOf = [&](const char *Port) -> int {
      for (auto &P : Inputs_[I])
        if (P.DstPort == Port) return static_cast<int>(P.SrcBlock);
      return -1;
    };
    int ClkSrc = srcOf("clk");
    int RstSrc = srcOf("reset");
    if (RstSrc < 0) RstSrc = srcOf("rst");
    double &Q = DigitalLatch_[I];
    if (RstSrc >= 0 && Out_[RstSrc] > 0.5) {
      Q = paramD(M_.Blocks[I], "initialValue", 0.0);
      continue;
    }
    bool Posedge =
        ClkSrc >= 0 && PrevOut_[ClkSrc] <= 0.0 && Out_[ClkSrc] > 0.0;
    if (!Posedge) continue;
    if (K == "signal_dff") {
      int DSrc = srcOf("d");
      if (DSrc < 0) DSrc = srcOf("in");
      if (DSrc < 0) DSrc = srcOf("in1");
      Q = (DSrc >= 0) ? Out_[DSrc] : 0.0;
    } else if (K == "signal_tff") {
      int TSrc = srcOf("t");
      if (TSrc < 0) TSrc = srcOf("in");
      bool Toggle = TSrc < 0 || Out_[TSrc] > 0.5;
      if (Toggle) Q = (Q > 0.5) ? 0.0 : 1.0;
    } else if (K == "signal_counter") {
      Q += paramD(M_.Blocks[I], "step", 1.0);
      double Mod = paramD(M_.Blocks[I], "modulus", 0.0);
      if (Mod > 0.0 && Q >= Mod) Q -= Mod;
    } else if (K == "signal_jkff") {
      // JK flip-flop: (J,K) → 00 hold, 01 reset, 10 set, 11 toggle.
      int JSrc = srcOf("j");
      if (JSrc < 0) JSrc = srcOf("in1");
      int KSrc = srcOf("k");
      if (KSrc < 0) KSrc = srcOf("in2");
      bool J = JSrc >= 0 && Out_[JSrc] > 0.5;
      bool Kk = KSrc >= 0 && Out_[KSrc] > 0.5;
      bool Qh = Q > 0.5;
      if (J && Kk)       Q = Qh ? 0.0 : 1.0; // toggle
      else if (J)        Q = 1.0;            // set
      else if (Kk)       Q = 0.0;            // reset
      // else hold
    } else { // signal_srff
      // SR flip-flop: (S,R) → 10 set, 01 reset, 00 hold, 11 hold (invalid).
      int SSrc = srcOf("s");
      if (SSrc < 0) SSrc = srcOf("in1");
      int RSrc2 = srcOf("r");
      if (RSrc2 < 0) RSrc2 = srcOf("in2");
      bool S = SSrc >= 0 && Out_[SSrc] > 0.5;
      bool R = RSrc2 >= 0 && Out_[RSrc2] > 0.5;
      if (S && !R)      Q = 1.0;
      else if (R && !S) Q = 0.0;
      // S==R: hold (the 1,1 case is undefined in HW; we hold)
    }
  }
  // #343 — HDL memory (signal_shift_register / signal_ram). On a `clk` posedge
  // the shift register marches every stage one step (serial `in` → stage 0),
  // and the RAM writes `data` at `addr` when `we` is high. ROM is stateless.
  // Active-high reset reloads the shift chain's initial value.
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const std::string &K = M_.Blocks[I].Kind;
    if (K != "signal_shift_register" && K != "signal_ram")
      continue;
    auto srcOf = [&](const char *Port) -> int {
      for (auto &P : Inputs_[I])
        if (P.DstPort == Port) return static_cast<int>(P.SrcBlock);
      return -1;
    };
    int ClkSrc = srcOf("clk");
    bool Posedge =
        ClkSrc >= 0 && PrevOut_[ClkSrc] <= 0.0 && Out_[ClkSrc] > 0.0;
    auto &Mem = HdlMem_[I];
    if (Mem.empty()) continue;
    if (K == "signal_shift_register") {
      int RstSrc = srcOf("reset");
      if (RstSrc < 0) RstSrc = srcOf("rst");
      if (RstSrc >= 0 && Out_[RstSrc] > 0.5) {
        std::fill(Mem.begin(), Mem.end(),
                  paramD(M_.Blocks[I], "initialValue", 0.0));
        continue;
      }
      if (!Posedge) continue;
      int InSrc = srcOf("in");
      if (InSrc < 0) InSrc = srcOf("in1");
      double NewBit = (InSrc >= 0) ? Out_[InSrc] : 0.0;
      for (int s = (int)Mem.size() - 1; s >= 1; --s) Mem[s] = Mem[s - 1];
      Mem[0] = NewBit;
    } else { // signal_ram
      if (!Posedge) continue;
      int WeSrc = srcOf("we");
      bool We = WeSrc < 0 || Out_[WeSrc] > 0.5; // no `we` wired ⇒ always write
      if (!We) continue;
      int AddrSrc = srcOf("addr");
      int Addr = (AddrSrc >= 0) ? (int)std::llround(Out_[AddrSrc]) : 0;
      if (Addr < 0) Addr = 0;
      if (Addr >= (int)Mem.size()) Addr = (int)Mem.size() - 1;
      int DataSrc = srcOf("data");
      if (DataSrc < 0) DataSrc = srcOf("in");
      Mem[Addr] = (DataSrc >= 0) ? Out_[DataSrc] : 0.0;
    }
  }
  // #343 — Communications error-rate (BER) sink. Once per major step, compare
  // the `tx`/`rx` inputs and accumulate the symbol/mismatch counts. Symbols
  // count as different when |tx - rx| exceeds half a level (default 0.5, or
  // `params.tolerance`), matching hard-decision BER on 0/1-valued streams.
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    if (M_.Blocks[I].Kind != "signal_error_rate")
      continue;
    auto srcOf = [&](const char *Port) -> int {
      for (auto &P : Inputs_[I])
        if (P.DstPort == Port) return static_cast<int>(P.SrcBlock);
      return -1;
    };
    int TxSrc = srcOf("tx");
    if (TxSrc < 0) TxSrc = srcOf("in1");
    if (TxSrc < 0) TxSrc = srcOf("in");
    int RxSrc = srcOf("rx");
    if (RxSrc < 0) RxSrc = srcOf("in2");
    if (TxSrc < 0 || RxSrc < 0)
      continue; // both inputs required to score a symbol
    double Tol = paramD(M_.Blocks[I], "tolerance", 0.5);
    TotAccum_[I] += 1.0;
    if (std::fabs(Out_[TxSrc] - Out_[RxSrc]) > Tol)
      ErrAccum_[I] += 1.0;
  }
  // #343 — streaming statistics. Once per major step, fold the current input
  // into each running_stats block's Welford accumulator (numerically stable
  // online mean/variance).
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    if (M_.Blocks[I].Kind != "signal_running_stats")
      continue;
    int Src = -1;
    for (auto &P : Inputs_[I])
      if (P.DstPort == "in" || P.DstPort == "in1") {
        Src = static_cast<int>(P.SrcBlock);
        break;
      }
    if (Src < 0)
      continue;
    double X = Out_[Src];
    RunCount_[I] += 1.0;
    double Delta = X - RunMean_[I];
    RunMean_[I] += Delta / RunCount_[I];
    RunM2_[I] += Delta * (X - RunMean_[I]);
  }
  // #343 — discrete Kalman filter. Once per major step, fold the measurement
  // (and optional control) into the standard predict/update recursion:
  //   predict: x⁻ = A·x + B·u ;  P⁻ = A·P·Aᵀ + Q
  //   update:  S = C·P⁻·Cᵀ + R ;  K = P⁻·Cᵀ·S⁻¹
  //            x = x⁻ + K·(z − C·x⁻) ;  P = (I − K·C)·P⁻
  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    if (M_.Blocks[I].Kind != "signal_kalman")
      continue;
    KalmanState &KS = Kalman_[I];
    if (!KS.Valid)
      continue;
    const int N = KS.N, Mz = KS.Mz;
    // Read a (possibly vector) input port into a fixed-width vector.
    auto readVec = [&](std::initializer_list<const char *> names,
                       int Want) -> std::vector<double> {
      std::vector<double> V(Want, 0.0);
      int Src = -1;
      for (const char *nm : names) {
        for (auto &P : Inputs_[I])
          if (P.DstPort == nm) { Src = static_cast<int>(P.SrcBlock); break; }
        if (Src >= 0) break;
      }
      if (Src < 0) return V;
      if (OutWidth_[Src] > 1)
        for (int e = 0; e < Want && e < (int)VecOut_[Src].size(); ++e)
          V[e] = VecOut_[Src][e];
      else
        V[0] = Out_[Src];
      return V;
    };
    std::vector<double> z = readVec({"z", "measurement", "in1", "in"}, Mz);
    // Predict. x⁻ = A·x (+ B·u)
    std::vector<double> xPred = matMul(KS.A, N, N, KS.X, N, 1);
    if (KS.P > 0) {
      std::vector<double> u = readVec({"u", "control", "in2"}, KS.P);
      std::vector<double> Bu = matMul(KS.B, N, KS.P, u, KS.P, 1);
      matAddInto(xPred, Bu);
    }
    // P⁻ = A·P·Aᵀ + Q
    std::vector<double> AP = matMul(KS.A, N, N, KS.Pc, N, N);
    std::vector<double> At = matT(KS.A, N, N);
    std::vector<double> Ppred = matMul(AP, N, N, At, N, N);
    matAddInto(Ppred, KS.Q);
    // S = C·P⁻·Cᵀ + R
    std::vector<double> Ct = matT(KS.C, Mz, N);
    std::vector<double> CP = matMul(KS.C, Mz, N, Ppred, N, N);
    std::vector<double> S = matMul(CP, Mz, N, Ct, N, Mz);
    matAddInto(S, KS.R);
    std::vector<double> Sinv;
    if (!matInv(S, Mz, Sinv)) {
      // Singular innovation covariance — skip the update, keep the prediction.
      KS.X = xPred;
      KS.Pc = Ppred;
      continue;
    }
    // K = P⁻·Cᵀ·S⁻¹   (N×Mz)
    std::vector<double> PCt = matMul(Ppred, N, N, Ct, N, Mz);
    std::vector<double> Kg = matMul(PCt, N, Mz, Sinv, Mz, Mz);
    // innovation y = z − C·x⁻
    std::vector<double> Cx = matMul(KS.C, Mz, N, xPred, N, 1);
    std::vector<double> y(Mz, 0.0);
    for (int e = 0; e < Mz; ++e) y[e] = z[e] - Cx[e];
    // x = x⁻ + K·y
    std::vector<double> Ky = matMul(Kg, N, Mz, y, Mz, 1);
    KS.X = xPred;
    matAddInto(KS.X, Ky);
    // P = (I − K·C)·P⁻
    std::vector<double> KC = matMul(Kg, N, Mz, KS.C, Mz, N); // N×N
    std::vector<double> ImKC(static_cast<size_t>(N) * N, 0.0);
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < N; ++c)
        ImKC[static_cast<size_t>(r) * N + c] =
            (r == c ? 1.0 : 0.0) - KC[static_cast<size_t>(r) * N + c];
    KS.Pc = matMul(ImKC, N, N, Ppred, N, N);
  }
  // Refresh outputs once more so the reset-into-state propagates
  // visibly into the post-step Out_ slot (the integrator's output
  // equals its state).
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
    } else if (isDiscreteIirKind(B.Kind)) {
      // §17.5 #5 — full direct-form-II IIR with u-history (FIR
      // numerator path). User provides num and den as polynomials
      // in z, highest order first. We rewrite as z^-1:
      //   H(z) = (Σ NumPad[k]·z^-k) / (Σ Den[k]·z^-k), k = 0..N
      //   where N = DenLen - 1 and NumPad pads Num with leading
      //   zeros to align with Den.
      // Time domain (after dividing by Den[0]):
      //   y[n] = (1/Den[0]) · ( Σ NumPad[k]·u[n-k] − Σ Den[k]·y[n-k] )
      // Z_ holds y[n-1]..y[n-(DenLen-1)]; FirHistory_ holds
      // u[n]..u[n-(NumLen-1+pad)].
      const auto &TF = TFCache_[I];
      int DenLen = static_cast<int>(TF.Den.size());
      int NumLen = static_cast<int>(TF.Num.size());
      auto &UHist = FirHistory_[I];
      if ((int)UHist.size() < std::max(DenLen, NumLen))
        UHist.resize(std::max(DenLen, NumLen), 0.0);
      // Shift u-history: UHist[K] = u[n-K]. Newest at index 0.
      for (int K = (int)UHist.size() - 1; K >= 1; --K)
        UHist[K] = UHist[K - 1];
      UHist[0] = U;

      double Y = 0.0;
      if (TF.Valid && DenLen >= 1) {
        double Lead = TF.Den.front();
        if (Lead == 0.0) Lead = 1.0;
        int N = DenLen - 1;
        // Padding offset: NumPad coefficient for z^-k corresponds
        // to TF.Num[k - UShift] when k ≥ UShift, else 0.
        int UShift = DenLen - NumLen;
        if (UShift < 0) UShift = 0;
        // Numerator sum: Σ TF.Num[j] · u[n - (UShift + j)].
        for (int J = 0; J < NumLen; ++J) {
          int HistIdx = UShift + J;
          double UVal = (HistIdx < (int)UHist.size())
                          ? UHist[HistIdx] : 0.0;
          Y += TF.Num[J] * UVal;
        }
        // Denominator feedback: − Σ TF.Den[k] · y[n-k], k = 1..N.
        for (int K = 1; K <= N; ++K) {
          Y -= TF.Den[K] * Z_[Off + K - 1];
        }
        Y /= Lead;
      } else {
        Y = U;
      }
      // Shift y-history.
      int N = DenLen - 1;
      for (int K = N - 1; K >= 1; --K) Z_[Off + K] = Z_[Off + K - 1];
      if (N >= 1) Z_[Off + 0] = Y;
      if (N == 0) Z_[Off] = Y;  // pure FIR (no feedback)
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

// RFC 4180 field encoding: a field that contains a comma, a double-quote,
// or a line break must be wrapped in double-quotes, with any embedded quote
// doubled. N-D / image column names carry comma-separated subscripts
// (`base[i,j]`), so without this the header is not parseable as CSV — a
// consumer that splits on `,` mis-counts fields (#392). Scalar / 1-D names
// (`base`, `base[k]`) have no comma and pass through unquoted.
static std::string csvField(const std::string &S) {
  if (S.find_first_of(",\"\n\r") == std::string::npos) return S;
  std::string Out = "\"";
  for (char C : S) {
    if (C == '"') Out += "\"\"";
    else Out += C;
  }
  Out += '"';
  return Out;
}

void MflowLinkSim::writeCsv(std::ostream &OS) const {
  OS << "t";
  for (auto &N : LogNames_) OS << "," << csvField(N);
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
  S.FirHistory = FirHistory_;
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
  FirHistory_ = std::move(S.FirHistory);
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

// #354 — source-line breakpoints inside MATLAB Function blocks.
void MflowLinkSim::setSourceBreakpoints(const std::string &BlockId,
                                        const std::vector<int> &Lines) {
  std::set<int> S(Lines.begin(), Lines.end());
  if (S.empty())
    SourceBreakpoints_.erase(BlockId);
  else
    SourceBreakpoints_[BlockId] = std::move(S);
}

MflowLinkSim::SourceHit MflowLinkSim::consumeSourceBreakpointHit() {
  SourceHit H = LastSourceHit_;
  LastSourceHit_ = SourceHit{};
  // #386 — a hit transitions the captured replay context into an active
  // stepping session, so the next/stepIn/stepOut requests can walk the body.
  if (H.Line >= 0 && PendingStep_.Valid) {
    FcnStep_ = PendingStep_;
    FcnStep_.Active = true;
    PendingStep_ = FcnStepState{};
  }
  return H;
}

MflowLinkSim::SourceHit MflowLinkSim::fcnStepNext() {
  // Re-run the (pure) body to the next statement index and surface its line +
  // locals. When the body completes before reaching that index, stepping ends.
  if (!FcnStep_.Active || !FcnStep_.Cache) return SourceHit{};
  int Target = FcnStep_.StmtIndex + 1;
  int Line = -1, Idx = -1;
  std::map<std::string, double> Vars;
  std::vector<double> Inputs = FcnStep_.Inputs;
  runMatlabFunction(*FcnStep_.Cache, Inputs, FcnStep_.T,
                    /*WatchLines=*/nullptr, &Line, &Vars,
                    /*StopAtStmt=*/Target, &Idx);
  if (Line < 0) {
    // The body completed without reaching the target — stepping is done.
    FcnStep_.Active = false;
    FcnStep_.Valid = false;
    return SourceHit{};
  }
  FcnStep_.StmtIndex = Target;
  SourceHit H;
  H.BlockId = FcnStep_.BlockId;
  H.Line = Line;
  H.Vars = std::move(Vars);
  return H;
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
  // #344: a MATLAB Function block may declare several outputs
  // (`function [a, b] = f(...)`), bound positionally to ports out1..outM.
  std::vector<std::string> OutNames;
};

namespace {
unsigned matlabFunctionInputCount(
    const MflowLinkSim::MatlabFunctionState &S) {
  return static_cast<unsigned>(S.InNames.size());
}
unsigned matlabFunctionOutputCount(
    const MflowLinkSim::MatlabFunctionState &S) {
  return static_cast<unsigned>(S.OutNames.size());
}
} // namespace

MflowLinkSim::~MflowLinkSim() {
  // §17.5 #8 — release every JIT'd block handle. `Release` is the
  // factory's tear-down hook; it owns the ExecutionEngine + JIT'd
  // memory and is responsible for freeing both.
  if (MatlabFcnJitOps_.Release) {
    for (auto *H : MatlabFnJit_) {
      if (H) MatlabFcnJitOps_.Release(H);
    }
  }
}

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
  if (TU->Functions.empty())
    return {nullptr, "function_body must declare at least one function"};
  // §17.5 #8 — the first function is the entry; trailing functions
  // are treated as locals. Bodies with helpers work under the JIT
  // (where lowering wires through user calls properly) and also
  // under the AST interpreter so long as the entry doesn't actually
  // call the helpers at runtime (the interpreter only knows
  // builtins). Loader keeps the same first-function-is-entry rule
  // either way.
  S->Fn = TU->Functions.front();
  if (S->Fn->Outputs.empty())
    return {nullptr,
            "function_body's entry function must declare at least one "
            "output variable"};
  // #344: bind every declared output positionally to ports out1..outM
  // (mirroring the u1..uN input convention); a single output is the
  // common case and drives the scalar `out`.
  S->OutNames.reserve(S->Fn->Outputs.size());
  for (auto &Out : S->Fn->Outputs) S->OutNames.emplace_back(Out);
  S->InNames.reserve(S->Fn->Inputs.size());
  for (auto &In : S->Fn->Inputs) S->InNames.emplace_back(In);
  return {std::move(S), std::string{}};
}

//===-----------------------------------------------------------------===//
// Scalar AST interpreter for `signal_matlab_fcn` / function_body.
//===-----------------------------------------------------------------===//

struct InterpEnv {
  std::map<std::string, double> Vars;
  // #354 — source-line breakpoints inside a MATLAB Function block. When set,
  // interpStmt resolves each statement's body line via `SM` and, if it is in
  // `WatchLines`, records it in `*HitLine` (the first hit wins). Null in normal
  // (non-debug) runs, so the hook is zero-cost.
  const std::set<int> *WatchLines = nullptr;
  const matlab::SourceManager *SM = nullptr;
  int *HitLine = nullptr;
  // #354 — on the first armed-line hit, snapshot the body's locals here (the
  // values visible *before* that line executes), for the DAP "Locals" scope.
  std::map<std::string, double> *HitVars = nullptr;
  // #386 — statement stepping via deterministic replay. `StmtCounter` is the
  // execution-order index of the next statement; when it equals `StopAtStmt`,
  // interpStmt captures line + locals and unwinds (StepStop). `HitStmtIdx`
  // records the index at a breakpoint hit so stepping knows where it paused.
  int StmtCounter = 0;
  int StopAtStmt = -1;
  int *HitStmtIdx = nullptr;
};
struct ReturnSignal {};
// #386 — thrown to unwind the body interpreter at a step target.
struct StepStop {};
// §17.5 #8 — break / continue inside for / while bodies.
struct BreakSignal {};
struct ContinueSignal {};

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
  // #354/#386 — debug hook. Resolve this statement's body line, then:
  //  - step mode (StopAtStmt ≥ 0): stop *before* the target statement,
  //    capturing line + locals, and unwind (StepStop). Zero-cost otherwise.
  //  - breakpoint (WatchLines): record the first armed line + its statement
  //    index + locals; the body keeps running (stepping replays it afterward).
  if (Env.SM && S->Range.Begin.isValid() &&
      (Env.StopAtStmt >= 0 || Env.WatchLines)) {
    int Ln = static_cast<int>(Env.SM->getLineColumn(S->Range.Begin).Line);
    if (Env.StopAtStmt >= 0 && Env.StmtCounter == Env.StopAtStmt) {
      if (Env.HitLine) *Env.HitLine = Ln;
      if (Env.HitVars) *Env.HitVars = Env.Vars;
      throw StepStop{};
    }
    if (Env.WatchLines && Env.HitLine && *Env.HitLine < 0 &&
        Env.WatchLines->count(Ln)) {
      *Env.HitLine = Ln;
      if (Env.HitVars) *Env.HitVars = Env.Vars; // locals before this line runs
      if (Env.HitStmtIdx) *Env.HitStmtIdx = Env.StmtCounter;
    }
  }
  ++Env.StmtCounter;
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
  case MNK::ForStmt: {
    // §17.5 #8 — `for var = expr; body; end`. Supports the
    // numeric-range form (`for i = 1:n` or `for i = start:step:end`).
    // The induction variable is bound to each successive value in
    // Env.Vars[Var]. break / continue handled via sentinel
    // exceptions; return propagates as ReturnSignal.
    auto *FS = static_cast<const matlab::ForStmt *>(S);
    std::string Var(FS->Var);
    if (!FS->Iter) return;
    if (FS->Iter->Kind == MNK::RangeExpr) {
      auto *R = static_cast<const matlab::RangeExpr *>(FS->Iter);
      double Start = interpExpr(R->Start, Env);
      double End   = interpExpr(R->End, Env);
      double Step  = R->Step ? interpExpr(R->Step, Env) : 1.0;
      if (Step == 0.0) return;
      for (double V = Start; (Step > 0 ? V <= End + 1e-12
                                       : V >= End - 1e-12); V += Step) {
        Env.Vars[Var] = V;
        try {
          interpBlock(FS->Body, Env);
        } catch (const BreakSignal &) { break; }
          catch (const ContinueSignal &) { continue; }
      }
    } else {
      // Single-value form `for x = scalar`: one iteration with x = value.
      double V = interpExpr(FS->Iter, Env);
      Env.Vars[Var] = V;
      try { interpBlock(FS->Body, Env); }
      catch (const BreakSignal &) {}
      catch (const ContinueSignal &) {}
    }
    return;
  }
  case MNK::WhileStmt: {
    // §17.5 #8 — `while cond; body; end`. Cond is re-evaluated each
    // iteration. Same break / continue / return semantics as
    // ForStmt. Bounded at 1e6 iterations to keep infinite loops in
    // user code from hanging the simulator silently.
    auto *WS = static_cast<const matlab::WhileStmt *>(S);
    long Guard = 0;
    while (interpExpr(WS->Cond, Env) != 0.0) {
      if (++Guard > 1000000) break;
      try {
        interpBlock(WS->Body, Env);
      } catch (const BreakSignal &) { break; }
        catch (const ContinueSignal &) { continue; }
    }
    return;
  }
  case MNK::BreakStmt:    throw BreakSignal{};
  case MNK::ContinueStmt: throw ContinueSignal{};
  default:
    return;
  }
}

// #344: returns one value per declared output (out1..outM), in order. An
// output variable never assigned by the body reads back as 0.
std::vector<double> runMatlabFunction(const MflowLinkSim::MatlabFunctionState &S,
                                      const std::vector<double> &Inputs,
                                      double T,
                                      const std::set<int> *WatchLines,
                                      int *HitLine,
                                      std::map<std::string, double> *HitVars,
                                      int StopAtStmt, int *HitStmtIdx) {
  InterpEnv Env;
  Env.Vars["t"] = T;
  for (size_t I = 0; I < S.InNames.size(); ++I)
    Env.Vars[S.InNames[I]] = I < Inputs.size() ? Inputs[I] : 0.0;
  // The shorthand `u` alias (matches our expression-evaluator
  // semantics — first input is `u`).
  if (!Inputs.empty()) Env.Vars["u"] = Inputs.front();
  // #354 — arm the source-line breakpoint hook for this body, if any.
  if (WatchLines && !WatchLines->empty() && HitLine) {
    Env.WatchLines = WatchLines;
    Env.SM = S.SM.get();
    Env.HitLine = HitLine;
    Env.HitVars = HitVars;
    Env.HitStmtIdx = HitStmtIdx;
  }
  // #386 — step mode is armed independently of breakpoints (replay stepping).
  if (StopAtStmt >= 0 && HitLine) {
    Env.SM = S.SM.get();
    Env.HitLine = HitLine;
    Env.HitVars = HitVars;
    Env.StopAtStmt = StopAtStmt;
  }
  try {
    interpBlock(S.Fn->Body, Env);
  } catch (const ReturnSignal &) {
    // Normal `return` exit.
  } catch (const StepStop &) {
    // #386 — body unwound at the step target; outputs are partial (unused —
    // stepping only inspects line + locals).
  }
  std::vector<double> Outs;
  Outs.reserve(S.OutNames.size());
  for (const auto &Name : S.OutNames) {
    auto It = Env.Vars.find(Name);
    Outs.push_back(It == Env.Vars.end() ? 0.0 : It->second);
  }
  return Outs;
}

} // namespace

std::string validateMatlabFunctionBody(const std::string &Source) {
  auto Pair = parseMatlabFunctionBody(Source);
  if (Pair.first) return std::string{};
  return Pair.second.empty() ? std::string("invalid function_body")
                              : Pair.second;
}

} // namespace matlab::flowchart
