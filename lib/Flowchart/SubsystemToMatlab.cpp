//===----------------------------------------------------------------------===//
// SubsystemToMatlab — mflowLink Embedded Coder, Tier 1.
//
// Walks the named Flow's subgraph in topological order and emits one
// MATLAB statement per block into a synthesised `matlab::Function`
// AST. The output feeds straight into the existing matlab_llvm
// `-emit-*` lanes. See `docs/embedded_coder_roadmap.md` §3 for the
// architecture rationale and §6 for the per-block lowering table.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/SubsystemToMatlab.h"

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace matlab::flowchart {

namespace {

//===-----------------------------------------------------------------===//
// String / identifier helpers.
//===-----------------------------------------------------------------===//

// Replace any character that's not [A-Za-z0-9_] with `_`; prepend `b_`
// if the result would start with a digit. Idempotent on well-formed
// identifiers.
std::string sanitizeIdent(const std::string &S) {
  std::string Out;
  Out.reserve(S.size());
  for (char C : S) {
    if (std::isalnum((unsigned char)C) || C == '_') Out.push_back(C);
    else Out.push_back('_');
  }
  if (Out.empty() || std::isdigit((unsigned char)Out.front()))
    Out = "b_" + Out;
  return Out;
}

// Parse a port id of the shape `u3` / `y2` / `in1` / `out2` → integer.
// Returns -1 when the id doesn't follow the convention; the caller
// then falls back to the textual sort.
int parsePortIndex(const std::string &Id) {
  size_t I = 0;
  while (I < Id.size() && !std::isdigit((unsigned char)Id[I])) ++I;
  if (I >= Id.size()) return -1;
  try { return std::stoi(Id.substr(I)); } catch (...) { return -1; }
}

//===-----------------------------------------------------------------===//
// Tier-1 supported block kinds.
//===-----------------------------------------------------------------===//
const std::set<std::string> &tier1Kinds() {
  static const std::set<std::string> S = {
      "signal_constant", "signal_gain",         "signal_sum",
      "signal_product",  "signal_abs",          "signal_saturation",
      "signal_math_fcn", "signal_trig_fcn",     "signal_relop",
      "signal_logical",  "signal_compare_to_zero",
      "signal_compare_to_constant",
      "signal_mux",      "signal_demux",        "signal_reshape",
      "signal_switch",   "signal_multiport_switch",
      "signal_merge",
      // Tier 3 — stateful discrete blocks.  Each contributes one
      // scalar state slot to the emitted function's signature
      // (one extra arg + one extra return).  See the `isStatefulKind`
      // helper below for the actual list and `stateSlotName` for the
      // naming convention.
      "signal_unit_delay", "signal_zoh",
      "signal_discrete_integrator",
      // Tier 4 — continuous-time integrator, auto-discretised at
      // the subsystem's `Ts` (CLI --target-rate / block sample_time
      // / settings.solver.maxStep). Rejected with a sourced error
      // when SubsystemEmitOptions.RejectContinuous = true (HDL lane).
      "signal_integrator",
      // Tier 5 — inline user MATLAB. The block's `params.function_body`
      // becomes a sibling local function in the same TU; the call
      // site emits as `<out> = <fn_name>(<inputs...>)`. SV emit
      // delegates synthesisability to the existing -check-synthesizable
      // pass over the user body.
      "signal_matlab_fcn",
      // Routing-only — emit nothing, but pass through the variable
      // name from the source.
      "signal_inport",   "signal_outport",
      // Sinks: drop entirely (scope/display/to_workspace/terminator
      // aren't part of the subsystem API).
      "signal_scope",    "signal_display",
      "signal_to_workspace", "signal_terminator",
  };
  return S;
}

bool isSinkKind(const std::string &K) {
  return K == "signal_scope" || K == "signal_display" ||
         K == "signal_to_workspace" || K == "signal_terminator";
}

// Tier 3 — stateful (loop-breaker) blocks. Each carries one state
// slot in the emitted function's signature: an extra scalar arg
// `s_<id>` (current state) and an extra scalar return
// `s_<id>_next` (state for the next tick). The state read happens
// at the start of the block's evaluation; the state-update lands
// in the next-state output.
//
// Tier 4 promotes `signal_integrator` (continuous-time `dx/dt = u`)
// into the stateful set: it gets auto-discretised to Forward Euler
// (or Backward Euler / Trapezoidal — see SubsystemEmitOptions) at
// the subsystem's chosen `Ts`. The lowering reports a sourced error
// if no rate can be resolved.
bool isStatefulKind(const std::string &K) {
  return K == "signal_unit_delay" ||
         K == "signal_zoh" ||
         K == "signal_discrete_integrator" ||
         K == "signal_integrator";
}

// Tier 4 — initial-condition param lookup per stateful kind.
// signal_integrator / signal_discrete_integrator carry an
// `initialCondition` (alt-spelling `initial_condition`). Unit Delay
// uses `initialCondition` too; ZOH uses `initialOutput`. Default 0.0.
double initialStateOf(const Node &N) {
  auto get = [&](const char *Key) -> const std::string * {
    auto It = N.Params.find(Key);
    return It == N.Params.end() ? nullptr : &It->second;
  };
  const std::string *S = get("initialCondition");
  if (!S) S = get("initial_condition");
  if (!S) S = get("initialOutput");
  if (!S) S = get("initial_value");
  if (!S) return 0.0;
  try { return std::stod(*S); } catch (...) { return 0.0; }
}

//===-----------------------------------------------------------------===//
// Per-edge lookup: for a given (Block, PortName), find the
// upstream Node id + its output variable name.
//===-----------------------------------------------------------------===//
struct EdgeIndex {
  // map: (toNodeId, toPortId) -> (fromNodeId, fromPortId)
  std::map<std::pair<std::string, std::string>,
           std::pair<std::string, std::string>> Map;
};
EdgeIndex buildEdgeIndex(const Flow &F) {
  EdgeIndex EI;
  for (const auto &E : F.Edges) {
    EI.Map[{E.To.Node, E.To.Port}] = {E.From.Node, E.From.Port};
  }
  return EI;
}

//===-----------------------------------------------------------------===//
// Topological sort over the Flow's edges. Skips inport / outport / sink
// kinds — they're handled separately. Returns the topo order of the
// internal blocks (those that emit a MATLAB statement).
//===-----------------------------------------------------------------===//
std::vector<const Node *>
toposortInternals(const Flow &F, DiagnosticEngine &Diag,
                  const std::string &SubName) {
  std::unordered_map<std::string, const Node *> NodeById;
  for (const auto &N : F.Nodes) NodeById[N.Id] = &N;

  // Internal nodes = neither inport nor outport nor sink.
  std::vector<const Node *> Internal;
  for (const auto &N : F.Nodes) {
    if (N.Kind == "signal_inport" || N.Kind == "signal_outport" ||
        isSinkKind(N.Kind))
      continue;
    Internal.push_back(&N);
  }

  // Build adjacency: in-edges per node, ignoring edges from sinks
  // (which won't appear in practice) and edges to outports / sinks
  // (those are terminal consumers).
  std::unordered_map<std::string, std::vector<std::string>> InEdges;
  std::unordered_map<std::string, int> InDegree;
  std::unordered_set<std::string> InternalSet;
  for (auto *N : Internal) InternalSet.insert(N->Id);

  // Loop-breaker rule: stateful blocks (Tier 3 + Tier 4 stateful set)
  // emit their CURRENT state as the output — that doesn't depend on
  // the in-this-tick input. Drop their outgoing edges from the topo
  // graph so feedback paths through `signal_integrator` /
  // `signal_unit_delay` / `signal_zoh` / `signal_discrete_integrator`
  // resolve cleanly. (The next-state update is computed *after* every
  // dependent block has consumed the current-state read, so the
  // back-edge is legitimately broken.)
  auto isLoopBreaker = [&](const std::string &NodeId) {
    auto It = std::find_if(F.Nodes.begin(), F.Nodes.end(),
                            [&](const Node &N) { return N.Id == NodeId; });
    if (It == F.Nodes.end()) return false;
    return isStatefulKind(It->Kind);
  };
  for (const auto &E : F.Edges) {
    if (!InternalSet.count(E.To.Node)) continue;
    if (!InternalSet.count(E.From.Node)) continue; // inport / sink-edge
    if (isLoopBreaker(E.From.Node)) continue;       // §6.3 loop-breaker
    InEdges[E.To.Node].push_back(E.From.Node);
    InDegree[E.To.Node]++;
  }

  // Kahn's algorithm. Initialise the queue with every internal node
  // that has zero in-edges from another internal node.
  std::vector<std::string> Ready;
  for (auto *N : Internal) {
    if (InDegree.find(N->Id) == InDegree.end()) Ready.push_back(N->Id);
  }
  // Stable order by node id so the emit output is deterministic.
  std::sort(Ready.begin(), Ready.end());

  std::vector<const Node *> Out;
  while (!Ready.empty()) {
    auto Id = Ready.front();
    Ready.erase(Ready.begin());
    Out.push_back(NodeById[Id]);
    // Find every internal node whose in-edges include Id and decrement.
    for (auto *N : Internal) {
      auto It = InEdges.find(N->Id);
      if (It == InEdges.end()) continue;
      auto &Sources = It->second;
      auto SIt = std::find(Sources.begin(), Sources.end(), Id);
      if (SIt == Sources.end()) continue;
      Sources.erase(SIt);
      if (--InDegree[N->Id] == 0) {
        Ready.push_back(N->Id);
        std::sort(Ready.begin(), Ready.end());
      }
    }
  }
  if (Out.size() != Internal.size()) {
    Diag.error(F.Loc, "subsystem \"" + SubName +
                          "\": cycle in subgraph — embedded-coder Tier-1 "
                          "does not solve algebraic loops");
    return {};
  }
  return Out;
}

//===-----------------------------------------------------------------===//
// AST builders — small helpers that wrap ASTContext::make.
//===-----------------------------------------------------------------===//
struct ASTBuilder {
  ASTContext &Ctx;
  // Tier 5 — when WrapFi is true, `lit(V)` wraps numeric literals in
  // `fi(V, signed, W, F)` so the SV pipeline sees concrete fixed-
  // point constants instead of f64.  Software-target emit leaves
  // the literal as a bare FPLiteral.
  bool WrapFi = false;
  bool FiSigned = true;
  int FiWidth = 32;
  int FiFrac = 16;

  NameExpr *name(const std::string &S) {
    auto *N = Ctx.make<NameExpr>();
    N->Name = Ctx.intern(S);
    return N;
  }
  FPLiteral *number(double V) {
    auto *F = Ctx.make<FPLiteral>();
    std::ostringstream OS;
    OS.precision(17);
    OS << V;
    F->Text = Ctx.intern(OS.str());
    return F;
  }
  IntegerLiteral *integer(int V) {
    auto *I = Ctx.make<IntegerLiteral>();
    I->Text = Ctx.intern(std::to_string(V));
    return I;
  }
  // Numeric literal honouring WrapFi — use this for ALL constant
  // operands inside lowerBlock's per-kind dispatch so a single flip
  // of WrapFi switches the entire emitter between software (raw
  // f64) and HDL (fi-wrapped) modes.
  Expr *lit(double V) {
    if (!WrapFi) return number(V);
    auto *Call = Ctx.make<CallOrIndex>();
    Call->Callee = name("fi");
    Call->Args.push_back(number(V));
    Call->Args.push_back(integer(FiSigned ? 1 : 0));
    Call->Args.push_back(integer(FiWidth));
    Call->Args.push_back(integer(FiFrac));
    return Call;
  }
  BinaryOpExpr *bin(BinOp Op, Expr *L, Expr *R) {
    auto *B = Ctx.make<BinaryOpExpr>();
    B->Op = Op; B->LHS = L; B->RHS = R;
    return B;
  }
  UnaryOpExpr *unary(UnOp Op, Expr *V) {
    auto *U = Ctx.make<UnaryOpExpr>();
    U->Op = Op; U->Operand = V;
    return U;
  }
  CallOrIndex *call(const std::string &FnName, std::vector<Expr *> Args) {
    auto *C = Ctx.make<CallOrIndex>();
    C->Callee = name(FnName);
    C->Args = std::move(Args);
    return C;
  }
  AssignStmt *assign(const std::string &Var, Expr *RHS) {
    auto *A = Ctx.make<AssignStmt>();
    A->LHS.push_back(name(Var));
    A->RHS = RHS;
    A->Suppressed = true;
    return A;
  }
};

//===-----------------------------------------------------------------===//
// Per-block lowering — the heart of Tier 1.
//
// Returns the AssignStmt that defines `<varName> = <expr>` for this
// block. The caller knows `varName` (the canonical output variable
// for the block) and `inputExprs` (one Expr per input port, sorted
// by port id). Returns nullptr if the block kind isn't supported.
//===-----------------------------------------------------------------===//
const std::string *paramS(const Node &N, const std::string &K) {
  return N.getParam(K);
}
double paramD(const Node &N, const std::string &K, double Def) {
  auto *S = paramS(N, K);
  if (!S) return Def;
  try { return std::stod(*S); } catch (...) { return Def; }
}
std::string paramSTR(const Node &N, const std::string &K,
                     const std::string &Def) {
  auto *S = paramS(N, K);
  return S ? *S : Def;
}

Stmt *lowerBlock(const Node &N, const std::string &OutVar,
                 const std::vector<Expr *> &Ins,
                 const std::vector<std::string> &InPortIds,
                 ASTBuilder &B, DiagnosticEngine &Diag) {
  const auto &K = N.Kind;

  auto get = [&](size_t I) -> Expr * {
    return I < Ins.size() ? Ins[I] : static_cast<Expr *>(B.lit(0.0));
  };

  if (K == "signal_constant") {
    // c = <value>;   (value can be scalar or matrix literal)
    auto V = paramSTR(N, "value", "0");
    // Vector / matrix literal: pass through as a MatrixLiteral. For
    // Tier 1 we parse only the scalar case; matrix literals route to
    // `signal_constant` via signal_reshape downstream.
    if (V.find('[') == std::string::npos) {
      double D = 0.0;
      try { D = std::stod(V); } catch (...) {}
      return B.assign(OutVar, B.lit(D));
    }
    // Strip the brackets, parse row-major numbers, build a MatrixLiteral.
    auto L = V.find('['), R = V.rfind(']');
    std::string Inner = V.substr(L + 1, R - L - 1);
    auto *ML = B.Ctx.make<MatrixLiteral>();
    std::string Tok;
    std::vector<Expr *> Row;
    auto endTok = [&]() {
      if (Tok.empty()) return;
      try { Row.push_back(B.lit(std::stod(Tok))); }
      catch (...) { Row.push_back(B.lit(0.0)); }
      Tok.clear();
    };
    auto endRow = [&]() {
      if (!Row.empty()) ML->Rows.push_back(std::move(Row));
      Row.clear();
    };
    for (size_t I = 0; I <= Inner.size(); ++I) {
      char C = I < Inner.size() ? Inner[I] : ';';
      if (C == ';') { endTok(); endRow(); }
      else if (C == ',' || C == ' ' || C == '\t') { endTok(); }
      else Tok.push_back(C);
    }
    return B.assign(OutVar, ML);
  }
  if (K == "signal_gain") {
    double Gain = paramD(N, "gain", 1.0);
    // y = K .* u
    auto *G = B.lit(Gain);
    auto *U = get(0);
    auto *Mul = B.bin(BinOp::ElemMul, G, U);
    return B.assign(OutVar, Mul);
  }
  if (K == "signal_sum") {
    // signs string like "+-+"; default = all '+' to match the input
    // count.
    auto Signs = paramSTR(N, "signs", std::string(Ins.size(), '+'));
    while (Signs.size() < Ins.size()) Signs.push_back('+');
    Expr *Acc = nullptr;
    for (size_t I = 0; I < Ins.size(); ++I) {
      Expr *T = Ins[I];
      bool Neg = (I < Signs.size() && Signs[I] == '-');
      if (!Acc) {
        Acc = Neg ? static_cast<Expr *>(B.unary(UnOp::Minus, T)) : T;
      } else {
        Acc = B.bin(Neg ? BinOp::Sub : BinOp::Add, Acc, T);
      }
    }
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_product") {
    Expr *Acc = nullptr;
    for (Expr *T : Ins) {
      Acc = Acc ? B.bin(BinOp::ElemMul, Acc, T) : T;
    }
    if (!Acc) Acc = B.lit(1.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_abs") {
    return B.assign(OutVar, B.call("abs", {get(0)}));
  }
  if (K == "signal_saturation") {
    double Lo = paramD(N, "lowerLimit", -1.0);
    double Hi = paramD(N, "upperLimit",  1.0);
    auto *U = get(0);
    // Tier-5d — HDL mode emits the if/elseif/else form. The pure-
    // arith form below (used for software targets) routes through
    // `bool * fi` multiplications that the SV synthcheck rejects;
    // an explicit if/elseif/else compiles to a clean 3-way mux at
    // synth time.
    if (B.WrapFi) {
      // Simple if/elseif/else form. All three branches store the
      // upstream-typed value (rails are fi-typed literals; the
      // else-branch passthrough carries the value verbatim).
      // For stateless subsystems this works cleanly. For subsystems
      // where the upstream chain widens to i64 (e.g. a discrete-PID
      // accumulator chain through fi-saturate), the alloc slot
      // ends up with mixed-width stores and HWLegalize rejects.
      // Pre-coercing all branches via `+ fi(0, ...)` widens the
      // rails to match — but breaks the stateless case (where
      // `c + 0` triggers a fi-saturate that the variable
      // passthrough doesn't take). Settling for the simple form;
      // PID-with-saturation needs Tier-5e (a dedicated MLIR pass
      // that detects mixed-width stores and unifies them).
      auto *IfStmt = B.Ctx.make<class IfStmt>();
      IfStmt->Cond = B.bin(BinOp::Gt, U, B.lit(Hi));
      IfStmt->Then = B.Ctx.make<Block>();
      IfStmt->Then->Stmts.push_back(B.assign(OutVar, B.lit(Hi)));
      ElseIf EI;
      EI.Cond = B.bin(BinOp::Lt, U, B.lit(Lo));
      EI.Body = B.Ctx.make<Block>();
      EI.Body->Stmts.push_back(B.assign(OutVar, B.lit(Lo)));
      IfStmt->Elseifs.push_back(EI);
      IfStmt->Else = B.Ctx.make<Block>();
      IfStmt->Else->Stmts.push_back(B.assign(OutVar, U));
      return IfStmt;
    }
    // Pure-arith form for software targets:
    //   y = u + (Hi - u) * (u > Hi) + (Lo - u) * (u < Lo)
    // Two correction terms; exactly one fires outside the rails.
    auto *DHi  = B.bin(BinOp::Sub, B.lit(Hi), U);
    auto *GtHi = B.bin(BinOp::Gt,  U, B.lit(Hi));
    auto *CHi  = B.bin(BinOp::ElemMul, DHi, GtHi);
    auto *DLo  = B.bin(BinOp::Sub, B.lit(Lo), U);
    auto *LtLo = B.bin(BinOp::Lt,  U, B.lit(Lo));
    auto *CLo  = B.bin(BinOp::ElemMul, DLo, LtLo);
    auto *S1   = B.bin(BinOp::Add, U, CHi);
    auto *Sat  = B.bin(BinOp::Add, S1, CLo);
    return B.assign(OutVar, Sat);
  }
  if (K == "signal_math_fcn" || K == "signal_trig_fcn") {
    // `function` param picks the builtin: e.g. "sin", "cos", "exp",
    // "log", "sqrt", "pow"…
    auto Fn = paramSTR(N, "function", K == "signal_trig_fcn" ? "sin" : "exp");
    std::vector<Expr *> Args;
    for (Expr *T : Ins) Args.push_back(T);
    if (Args.empty()) Args.push_back(B.lit(0.0));
    return B.assign(OutVar, B.call(Fn, std::move(Args)));
  }
  if (K == "signal_relop") {
    auto Op = paramSTR(N, "op", "==");
    BinOp BO = BinOp::Eq;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), get(1)));
  }
  if (K == "signal_logical") {
    auto Op = paramSTR(N, "op", "and");
    Expr *Acc = nullptr;
    for (Expr *T : Ins) Acc = Acc ? B.bin(Op == "or"
                                              ? BinOp::Or
                                              : BinOp::And, Acc, T)
                                  : T;
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_compare_to_zero") {
    auto Op = paramSTR(N, "op", ">");
    BinOp BO = BinOp::Gt;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), B.lit(0.0)));
  }
  if (K == "signal_compare_to_constant") {
    auto Op = paramSTR(N, "op", ">");
    double C = paramD(N, "constant", 0.0);
    BinOp BO = BinOp::Gt;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), B.lit(C)));
  }
  if (K == "signal_mux") {
    // y = [in1, in2, ...]
    auto *ML = B.Ctx.make<MatrixLiteral>();
    std::vector<Expr *> Row;
    for (Expr *T : Ins) Row.push_back(T);
    ML->Rows.push_back(std::move(Row));
    return B.assign(OutVar, ML);
  }
  if (K == "signal_reshape") {
    int R = (int)paramD(N, "rows", 0.0);
    int C = (int)paramD(N, "cols", 0.0);
    if (R <= 0 || C <= 0) {
      // No shape declared — fall back to passthrough.
      return B.assign(OutVar, get(0));
    }
    return B.assign(OutVar,
                    B.call("reshape", {get(0), B.integer(R), B.integer(C)}));
  }
  if (K == "signal_demux") {
    // Tier-C semantics: passthrough first input.
    return B.assign(OutVar, get(0));
  }
  if (K == "signal_switch") {
    // Natural form `y = (ctrl > th)*in1 + (ctrl <= th)*in3` makes
    // the static -emit-* pipeline mis-infer the function's return
    // type as `bool` (the comparison's type leaks through the
    // multiplication when neither operand is a concrete f64
    // literal — saturation works because `Hi - u` anchors the
    // expression to `double` first). Anchor with a `0.0 +` so the
    // top-level Add is `double + (bool*double + bool*double)`:
    //
    //   y = 0.0 + (ctrl > threshold)*in1 + (ctrl <= threshold)*in3
    double Th = paramD(N, "threshold", 0.0);
    auto *Gt = B.bin(BinOp::Gt, get(1), B.lit(Th));
    auto *Le = B.bin(BinOp::Le, get(1), B.lit(Th));
    auto *T  = B.bin(BinOp::ElemMul, Gt, get(0));
    auto *F  = B.bin(BinOp::ElemMul, Le, get(2));
    auto *Sum = B.bin(BinOp::Add, T, F);
    auto *Anchored = B.bin(BinOp::Add, B.lit(0.0), Sum);
    return B.assign(OutVar, Anchored);
  }
  if (K == "signal_multiport_switch") {
    // y = data(in1) where in1 is the 1-based selector. For codegen,
    // emit a chained ternary via call to a synthetic `ms_switch` —
    // simpler: emit a multiport_switch as a sequence of if-else, but
    // the AST emit lanes don't lower IfStmt as expressions. For
    // Tier 1, restrict to the 2-input (selector + one data) form and
    // emit `data` (passthrough).
    return B.assign(OutVar, get(1));
  }
  if (K == "signal_merge") {
    // First non-zero input wins. Tier-1 simplification: emit a sum
    // of all inputs (works when only one is active at a time).
    Expr *Acc = nullptr;
    for (Expr *T : Ins) Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_matlab_fcn") {
    // Tier 5 — call into the user's inline MATLAB. The
    // `params.function_body` is parsed at the end of the main
    // lowering (so all matlab_fcn helpers are sibling functions in
    // the same TU); here we just emit the call site. Helper name:
    // `<userFnName>_<sanitizedBlockId>` — guarantees uniqueness if
    // two blocks happen to have user functions with the same name.
    auto *FB = N.getParam("function_body");
    if (!FB) {
      Diag.error(N.Loc, "signal_matlab_fcn \"" + N.Id +
                            "\": missing `params.function_body`");
      return nullptr;
    }
    // Extract function name (first identifier before `(`).
    std::string SourceTxt = *FB;
    size_t Pos = SourceTxt.find("function");
    size_t Eq  = SourceTxt.find('=', Pos);
    size_t Lp  = SourceTxt.find('(', Pos);
    std::string FnName;
    if (Pos != std::string::npos && Lp != std::string::npos) {
      size_t Start = (Eq != std::string::npos && Eq < Lp)
                         ? Eq + 1 : Pos + sizeof("function") - 1;
      while (Start < Lp &&
             (SourceTxt[Start] == ' ' || SourceTxt[Start] == '\t'))
        ++Start;
      size_t End = Lp;
      while (End > Start &&
             (SourceTxt[End - 1] == ' ' || SourceTxt[End - 1] == '\t'))
        --End;
      FnName = SourceTxt.substr(Start, End - Start);
    }
    if (FnName.empty()) {
      Diag.error(N.Loc, "signal_matlab_fcn \"" + N.Id +
                            "\": could not extract function name from "
                            "`params.function_body`");
      return nullptr;
    }
    // Renamed helper: <user_name>_<block_id>.
    std::string Helper = FnName + "_" + sanitizeIdent(N.Id);
    std::vector<Expr *> Args;
    for (Expr *T : Ins) Args.push_back(T);
    return B.assign(OutVar, B.call(Helper, std::move(Args)));
  }
  // Tier-1 doesn't cover this kind.
  Diag.error(N.Loc, "embedded coder Tier-1 doesn't yet support "
                    "`" + K + "` — see "
                    "docs/embedded_coder_roadmap.md §6 for the "
                    "block-coverage table");
  return nullptr;
}

//===-----------------------------------------------------------------===//
// Port collection — find inports / outports + sort by port index.
//===-----------------------------------------------------------------===//
struct PortInfo {
  const Node *N;
  int Index;   // 1-based from `u<k>` / `y<k>`, else order of appearance
  std::string Var;
};

std::vector<PortInfo> collectPorts(const Flow &F, const std::string &Kind) {
  std::vector<PortInfo> Out;
  for (const auto &N : F.Nodes) {
    if (N.Kind != Kind) continue;
    int Idx = parsePortIndex(N.Id);
    if (Idx < 0) Idx = (int)Out.size() + 1;
    Out.push_back({&N, Idx, ""});
  }
  std::sort(Out.begin(), Out.end(),
            [](const PortInfo &A, const PortInfo &B) {
              if (A.Index != B.Index) return A.Index < B.Index;
              return A.N->Id < B.N->Id;
            });
  // Default variable name = the node id sanitised; the caller renames
  // inports to `u<k>` / outports to `y<k>` for readable output.
  for (size_t I = 0; I < Out.size(); ++I) {
    if (Kind == "signal_inport")  Out[I].Var = "u" + std::to_string(I + 1);
    else if (Kind == "signal_outport") Out[I].Var = "y" + std::to_string(I + 1);
    else Out[I].Var = sanitizeIdent(Out[I].N->Id);
  }
  return Out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points.
//===----------------------------------------------------------------------===//

matlab::Function *lowerSubsystemToMatlab(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts) {
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (!Sub) {
    Diag.error(SourceLocation{}, "subsystem \"" + SubsystemName +
                                     "\" not found in `.mflow` file");
    return nullptr;
  }
  auto Inports = collectPorts(*Sub, "signal_inport");
  auto Outports = collectPorts(*Sub, "signal_outport");
  if (Inports.empty() && Outports.empty()) {
    Diag.error(Sub->Loc, "subsystem \"" + SubsystemName +
                             "\" has no signal_inport / signal_outport "
                             "boundary — top-level flows without ports "
                             "go through `-emit-mflowlink-cpp` instead");
    return nullptr;
  }

  ASTBuilder B{AST};
  // Tier 5 — flip the ASTBuilder into HDL mode so per-block numeric
  // literals emit as `fi(V, ...)` calls. SV synth-check rejects f64
  // operands; the static SV pipeline's `runLowerFixedPoint` resolves
  // fi-wrapped constants into the right integer width.
  if (Opts.StateAsPersistent) {
    B.WrapFi   = true;
    B.FiSigned = Opts.FiDefault.Signed;
    B.FiWidth  = Opts.FiDefault.Width;
    B.FiFrac   = Opts.FiDefault.Frac;
  }
  EdgeIndex EI = buildEdgeIndex(*Sub);

  // For each non-port block, decide a unique output variable name.
  std::unordered_map<std::string, std::string> VarOfNode;
  for (auto &P : Inports)  VarOfNode[P.N->Id] = P.Var;
  for (auto &P : Outports) VarOfNode[P.N->Id] = P.Var;
  // Avoid name collisions across blocks.
  std::set<std::string> Used;
  for (auto &Pr : VarOfNode) Used.insert(Pr.second);
  auto uniqueVarFor = [&](const std::string &Id) {
    std::string Base = sanitizeIdent(Id);
    if (Base == "u" || Base == "y" || Base == "t") Base = "v_" + Base;
    std::string Name = Base;
    int Suffix = 1;
    while (Used.count(Name)) Name = Base + "_" + std::to_string(++Suffix);
    Used.insert(Name);
    return Name;
  };

  auto Internal = toposortInternals(*Sub, Diag, SubsystemName);
  if (Internal.empty() && !Outports.empty()) {
    // Allow pure passthrough — an outport wired straight to an inport.
  }
  if (Diag.hasErrors()) return nullptr;

  // Tier 3 — collect stateful blocks in topo order; each contributes
  // one scalar state slot to the function signature: an extra arg
  // `s_<id>` (current state) and an extra return `s_<id>_next` (state
  // for the next tick). The state read is what the block's "output"
  // becomes; the next-state update lands in `s_<id>_next` at the end
  // of the body so the caller can latch it back for the next call.
  struct StateSlot {
    const Node *N;
    std::string CurArg;    // function arg name carrying current state
    std::string NextOut;   // function return name carrying next state
    std::string OutVar;    // local var = current state (block's output)
  };
  std::vector<StateSlot> States;
  for (auto *N : Internal) {
    if (!isStatefulKind(N->Kind)) continue;
    StateSlot S;
    S.N       = N;
    S.CurArg  = "s_" + sanitizeIdent(N->Id);
    S.NextOut = "s_" + sanitizeIdent(N->Id) + "_next";
    S.OutVar  = VarOfNode[N->Id];   // already populated below
    States.push_back(S);
  }
  // Pre-reserve all the state-related identifiers in `Used` so the
  // generic `uniqueVarFor` doesn't accidentally collide with them.
  for (auto &S : States) {
    Used.insert(S.CurArg);
    Used.insert(S.NextOut);
  }

  for (auto *N : Internal) {
    VarOfNode[N->Id] = uniqueVarFor(N->Id);
  }
  // Refresh OutVar now that VarOfNode is populated.
  for (auto &S : States) S.OutVar = VarOfNode[S.N->Id];

  // Build the function body.
  auto *Body = AST.make<Block>();

  // Resolve a Name expression to the variable feeding the named port.
  auto resolveInputExpr = [&](const std::string &ToNode,
                              const std::string &ToPort) -> Expr * {
    auto It = EI.Map.find({ToNode, ToPort});
    if (It == EI.Map.end()) return B.number(0.0);
    auto VarIt = VarOfNode.find(It->second.first);
    if (VarIt == VarOfNode.end()) return B.number(0.0);
    return B.name(VarIt->second);
  };

  // Find the input ports of a block (sorted by canonical name —
  // `in1`/`in2`/`u1`/`u2`/...).
  auto inputPortsOf = [&](const Node &N) -> std::vector<std::string> {
    std::vector<std::pair<int, std::string>> Pairs;
    for (const auto &P : N.InPorts) {
      int Idx = parsePortIndex(P.Id);
      if (Idx < 0) Idx = (int)Pairs.size() + 1;
      Pairs.push_back({Idx, P.Id});
    }
    std::sort(Pairs.begin(), Pairs.end());
    std::vector<std::string> Out;
    for (auto &PR : Pairs) Out.push_back(PR.second);
    return Out;
  };

  // §17.5-#4-style discretization spec for each stateful block —
  // computed up front so the per-block dispatch below can render
  // the `<next> = ...` update expression cleanly.
  std::unordered_map<std::string, Expr *> NextStateExpr;

  // Tier 5 — hard-reject continuous blocks for HDL emit. Software
  // targets (Tier 4) auto-discretise via `--target-rate`; SV must
  // be explicit (the user replaces `signal_integrator` with
  // `signal_discrete_integrator` and Unit Delays in the .mflow).
  if (Opts.RejectContinuous) {
    for (auto *N : Internal) {
      if (N->Kind == "signal_integrator" ||
          N->Kind == "signal_transfer_fcn" ||
          N->Kind == "signal_state_space" ||
          N->Kind == "signal_zero_pole" ||
          N->Kind == "signal_transport_delay") {
        Diag.error(N->Loc,
                   "HDL emit: continuous block `" + N->Kind + "` \"" +
                       N->Id + "\" can't be synthesised — replace with "
                       "the equivalent `signal_discrete_*` in the "
                       ".mflow source (the simulator's continuous "
                       "behaviour is software-only).");
        return nullptr;
      }
    }
  }

  // Tier 5 — `persistent <slot>; if isempty(<slot>) || reset; ... end`
  // initialisation block per stateful block, when SubsystemEmitOptions
  // wants HDL-style internal state (registers).  Routes through the
  // existing matlab_llvm SV pipeline's persistent → reg lowering
  // (`lib/MLIR/Passes/LowerPersistentFiArrays.cpp`).
  if (Opts.StateAsPersistent) {
    for (auto &S : States) {
      auto *Decl = AST.make<PersistentDecl>();
      Decl->Names.push_back(AST.intern(S.CurArg));
      Body->Stmts.push_back(Decl);
    }
    // Wrap each init under `if isempty(<slot>) || reset`.
    for (auto &S : States) {
      // Build: `isempty(<slot>) || reset` — short-circuit OR
      // (BinOp::ShortOr), the form the SV pipeline recognises in
      // its `if isempty(...)` init template.  Bitwise `|` would
      // route through arith.ori which requires integer operands.
      auto *IsEmpty = B.call("isempty", {B.name(S.CurArg)});
      auto *Or = B.bin(BinOp::ShortOr, IsEmpty, B.name("reset"));
      // Build the init expression. For HDL, route through `fi(...)`
      // so the persistent register gets the user's chosen format.
      // Lookup per-port spec; default to the global FiDefault.
      FixedPointSpec Spec = Opts.FiDefault;
      auto It = Opts.FiSpecs.find(S.CurArg);
      if (It != Opts.FiSpecs.end()) Spec = It->second;
      double InitVal = 0.0;
      // Pull the actual IC from the block — the StateSlot vector is
      // already populated above in topo order.
      for (const auto &N : Sub->Nodes) {
        if (N.Id == S.N->Id) {
          InitVal = initialStateOf(N);
          break;
        }
      }
      // fi(<value>, <signed>, <width>, <frac>)
      auto *FiCall = B.call("fi",
                            {B.number(InitVal),
                             B.integer(Spec.Signed ? 1 : 0),
                             B.integer(Spec.Width),
                             B.integer(Spec.Frac)});
      auto *Then = AST.make<Block>();
      Then->Stmts.push_back(B.assign(S.CurArg, FiCall));
      auto *If = AST.make<IfStmt>();
      If->Cond = Or;
      If->Then = Then;
      Body->Stmts.push_back(If);
    }
  }

  // Tier 3+4 — hoist every stateful block's STATE READ to the top
  // of the body, BEFORE any consumer evaluates. Without this hoist
  // the topo sort (which drops loop-breakers' outgoing edges) is
  // free to place the consumer ahead of the state-read assignment,
  // and the consumer would read the variable's pre-init zero value
  // — silently producing wrong outputs.  Matches the simulator's
  // "load Z_ first, then evalAll" tick shape (lib/Flowchart/
  // MflowLinkSim.cpp).
  for (auto *N : Internal) {
    if (!isStatefulKind(N->Kind)) continue;
    const std::string &OutV = VarOfNode[N->Id];
    const std::string CurS  = "s_" + sanitizeIdent(N->Id);
    // `<OutV> = <CurS> + 0.0;`  (output the latched state)
    // The `+ 0.0` anchors the assignment to `f64` so the static
    // -emit-* pipeline's slot-type inference picks `double`
    // throughout — without it a pure-passthrough subsystem (e.g.
    // one Unit Delay with no internal arithmetic) gets collapsed
    // away as dead-code and the emitted function body is empty.
    // The anchor is a no-op at runtime.  Persistent-mode reads
    // skip the anchor since the slot already carries a concrete
    // `fi(...)` type from the isempty-init block.
    if (Opts.StateAsPersistent) {
      Body->Stmts.push_back(B.assign(OutV, B.name(CurS)));
    } else {
      Body->Stmts.push_back(B.assign(OutV,
                                      B.bin(BinOp::Add, B.name(CurS),
                                            B.number(0.0))));
    }
  }

  // Emit one statement per internal block.
  for (auto *N : Internal) {
    auto Ports = inputPortsOf(*N);
    std::vector<Expr *> Ins;
    for (auto &P : Ports) Ins.push_back(resolveInputExpr(N->Id, P));

    if (isStatefulKind(N->Kind)) {
      // State read was hoisted above; we still need to compute the
      // next-state expression here (which DOES depend on the
      // consumer order because the integrator's input is `u[n]`,
      // i.e. the current-tick value upstream — which only exists
      // after every block feeding it has run).
      const std::string CurS  = "s_" + sanitizeIdent(N->Id);
      // Compute next state. Stash in `NextStateExpr` for the
      // final state-update block at end of body.
      Expr *NextExpr = nullptr;
      if (N->Kind == "signal_unit_delay" || N->Kind == "signal_zoh") {
        // Next state = current input. Software targets anchor with
        // `+ 0.0` so the static -emit-* pipeline's slot inference
        // picks `double` (otherwise pure-passthrough subsystems
        // get DCE'd to `pass`). HDL mode passes the input through
        // directly — adding an f64 literal would taint the
        // persistent slot's fi typing and trip the HWLegalize
        // synthcheck.
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        NextExpr = Opts.StateAsPersistent
                       ? U
                       : B.bin(BinOp::Add, U, B.number(0.0));
      } else if (N->Kind == "signal_discrete_integrator" ||
                 N->Kind == "signal_integrator") {
        // signal_integrator (continuous) gets auto-discretised here
        // — the math is identical to `signal_discrete_integrator`
        // once a sample rate is picked.  Forward-Euler-with-current-
        // u: s_next = s + Ts * u (matches the Tier-3 lowering).
        //
        // Ts resolution order:
        //   1. SubsystemEmitOptions.TargetRate (CLI --target-rate)
        //   2. block's `sample_time` / `sampleTime` / `Ts` param
        //   3. settings.solver.maxStep from the flow doc
        //   4. hard error
        double Ts = Opts.TargetRate;
        if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
        if (Ts <= 0.0) {
          // settings.solver.maxStep — only meaningful when it's a
          // numeric literal ("auto" means continuous).
          if (Doc.Settings.Solver.has_value()) {
            const auto &SC = *Doc.Settings.Solver;
            if (SC.MaxStep != "auto") {
              try { Ts = std::stod(SC.MaxStep); } catch (...) {}
            }
          }
        }
        if (Ts <= 0.0) {
          Diag.error(N->Loc,
                     "embedded coder: continuous `" + N->Kind +
                         "` block \"" + N->Id +
                         "\" needs a sample period — pass `--target-rate "
                         "<Ts>`, or set `data.sample_time` on the block, "
                         "or declare `settings.solver.maxStep` on the "
                         "flow");
          return nullptr;
        }
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        // Wrap the Ts coefficient in `fi(...)` when in HDL mode so
        // the multiplication stays in integer arithmetic. Software
        // targets emit the bare double (the IEEE arithmetic is what
        // matches the simulator's continuous integrator).
        Expr *TsConst = B.WrapFi ? B.lit(Ts) : B.number(Ts);
        Expr *TsU = B.bin(BinOp::ElemMul, TsConst, U);
        // Tier-5e — in HDL mode, reference the LOCAL state-read
        // variable (set by the hoisted state-read at top of body)
        // instead of the persistent slot. The local was already
        // converted from f64 → fi/i32 by fptosi; re-fetching the
        // persistent would yield f64 and trigger
        // `matlab.add(f64, i32) → none` downstream. Software
        // targets keep referencing the persistent slot directly
        // (the slot is a plain f64 var, no conversion needed).
        const std::string &LocalRead =
            B.WrapFi ? VarOfNode[N->Id] : CurS;
        NextExpr = B.bin(BinOp::Add, B.name(LocalRead), TsU);
      } else {
        NextExpr = B.number(0.0);
      }
      NextStateExpr[N->Id] = NextExpr;
      continue;
    }

    auto *Stmt = lowerBlock(*N, VarOfNode[N->Id], Ins, Ports, B, Diag);
    if (!Stmt) return nullptr;
    Body->Stmts.push_back(Stmt);
  }

  // For each outport, append `<port_var> = <feeding_var>;`.
  for (auto &P : Outports) {
    // An outport has exactly one input (`in`). Resolve to its source.
    Expr *Rhs = nullptr;
    for (const auto &Port : P.N->InPorts) {
      Rhs = resolveInputExpr(P.N->Id, Port.Id);
      break;
    }
    if (!Rhs) Rhs = B.number(0.0);
    Body->Stmts.push_back(B.assign(P.Var, Rhs));
  }

  // Tier 3 — emit `<NextOut> = <NextExpr>;` for every stateful block
  // so the multi-return picks up the next-state values. In Tier-5
  // persistent mode the next state lands directly in the persistent
  // slot (no separate `_next` return — the persistent itself is the
  // mutable storage), which the SV pipeline lowers to a register.
  for (auto &S : States) {
    Expr *E = NextStateExpr[S.N->Id];
    if (!E) E = B.number(0.0);
    if (Opts.StateAsPersistent) {
      Body->Stmts.push_back(B.assign(S.CurArg, E));
    } else {
      Body->Stmts.push_back(B.assign(S.NextOut, E));
    }
  }

  // Build the function node.
  auto *Fn = AST.make<Function>();
  Fn->Name = AST.intern(sanitizeIdent(SubsystemName));
  for (auto &P : Inports)  Fn->Inputs.push_back(AST.intern(P.Var));
  if (Opts.StateAsPersistent && !States.empty()) {
    // Tier 5 — HDL needs an explicit `reset` boundary to clear the
    // persistent regs on power-up. Add it as the final arg so the
    // synthesised SV module exposes `reset` alongside the data
    // inputs. Stateless subsystems have no regs and skip the reset.
    Fn->Inputs.push_back(AST.intern("reset"));
  } else if (!Opts.StateAsPersistent) {
    // Tier 3 — append state args after the inports so the public
    // signature reads `step(u1, ..., uN, s_<a>, s_<b>, ...)`.
    for (auto &S : States) Fn->Inputs.push_back(AST.intern(S.CurArg));
  }
  for (auto &P : Outports) Fn->Outputs.push_back(AST.intern(P.Var));
  if (!Opts.StateAsPersistent) {
    // And next-state returns after the regular outports:
    //   `[y1, ..., yM, s_<a>_next, s_<b>_next, ...]`.
    for (auto &S : States) Fn->Outputs.push_back(AST.intern(S.NextOut));
  }
  Fn->Body = Body;
  return Fn;
}

matlab::TranslationUnit *buildSubsystemTU(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts) {
  auto *Fn = lowerSubsystemToMatlab(Doc, SubsystemName, AST, Diag, Opts);
  if (!Fn) return nullptr;

  auto *TU = AST.make<TranslationUnit>();
  TU->Functions.push_back(Fn);

  // Tier 5 — collect every `signal_matlab_fcn` block in the subsystem
  // and add its `params.function_body` as a sibling local function in
  // the same TU. The block-level dispatch already emits a call site
  // named `<userFnName>_<sanitizedBlockId>` (renamed for uniqueness);
  // here we parse the user-supplied body, rename the entry, and
  // append to the TU's Functions list so Sema + lowering pick it up
  // alongside the main subsystem function.
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (Sub) {
    for (const auto &N : Sub->Nodes) {
      if (N.Kind != "signal_matlab_fcn") continue;
      auto It = N.Params.find("function_body");
      if (It == N.Params.end()) continue;
      // Parse the user body in its own SourceManager so its
      // SourceLocations don't collide with the main TU's. The
      // outer Diag still sees any errors.
      auto LocalSM = std::make_unique<matlab::SourceManager>();
      auto LocalDiag = std::make_unique<matlab::DiagnosticEngine>(*LocalSM);
      matlab::FileID F = LocalSM->addBuffer(
          "<signal_matlab_fcn:" + N.Id + ">", It->second);
      matlab::Lexer L(*LocalSM, F, *LocalDiag);
      auto Toks = L.tokenize();
      if (LocalDiag->hasErrors()) {
        Diag.error(N.Loc,
                   "signal_matlab_fcn \"" + N.Id +
                       "\": lex error in `params.function_body`");
        return nullptr;
      }
      matlab::Parser P(std::move(Toks), AST, *LocalDiag);
      auto *FnTU = P.parseFile();
      if (!FnTU || LocalDiag->hasErrors() || FnTU->Functions.empty()) {
        Diag.error(N.Loc,
                   "signal_matlab_fcn \"" + N.Id +
                       "\": parse error in `params.function_body`");
        return nullptr;
      }
      // Rename the entry function to `<orig>_<blockId>` so callers
      // who reference the unique helper name resolve cleanly.
      auto *UserFn = FnTU->Functions.front();
      std::string Helper =
          std::string(UserFn->Name) + "_" + sanitizeIdent(N.Id);
      UserFn->Name = AST.intern(Helper);
      TU->Functions.push_back(UserFn);
      // The LocalSM/LocalDiag go out of scope here; the AST nodes
      // they back stay alive in `AST` (the bump allocator on the
      // outer ASTContext). The string_views into the SM's buffer
      // would dangle — re-intern them.
      // (Best-effort: most string_view fields are short-lived
      // identifiers that the resolver re-interns; for fields like
      // FPLiteral.Text we walk the body and re-intern below.)
      // Simpler approach: intern the whole source text into `AST`
      // so the LocalSM's buffer isn't the backing store.  Since
      // we already have the text in `It->second`, we can ensure
      // it's interned in the outer AST too — but the parser
      // already created string_views pointing at the LocalSM
      // buffer. Drop both LocalSM and LocalDiag at TU end-of-life
      // by appending to a TU-scoped reservoir.  For now we leak
      // them deliberately by stashing in a static — they're tiny
      // per matlab_fcn block and the matlabc process exits at end
      // of compile anyway.
      static std::vector<std::unique_ptr<matlab::SourceManager>> KeepSM;
      static std::vector<std::unique_ptr<matlab::DiagnosticEngine>> KeepDiag;
      KeepSM.push_back(std::move(LocalSM));
      KeepDiag.push_back(std::move(LocalDiag));
    }
  }

  // Synthesise a driver script that calls the function with concrete
  // f64 args. That call site forces the static `-emit-*` pipeline to
  // refine the function's slots to `double` instead of `none` / `void*`
  // (the function-only-file pitfall documented during the §17.5 #8
  // work). The driver is a single AssignStmt placed in the TU's
  // Script node — matlab_llvm allows a script body before function
  // definitions in the same file.
  ASTBuilder B{AST};
  // Tier 5 — skip the priming driver for HDL emit. HDL gets its
  // arg/result types from the `hdl.ports` MLIR attribute the
  // caller stamps post-lowering (no need to type-refine via a
  // call site), and the call's f64 args confuse the SV pipeline
  // (e.g. `arith.shli` on an f64 operand of the constant-mul
  // optimisation).  Software targets keep the driver — they
  // need the concrete-typed call site to refine the function's
  // slots to `double`.
  if (!Fn->Inputs.empty() && !Opts.StateAsPersistent) {
    std::vector<Expr *> Args;
    for (size_t I = 0; I < Fn->Inputs.size(); ++I)
      Args.push_back(B.number(0.0));
    auto *Call = B.call(std::string(Fn->Name), std::move(Args));
    auto *Driver = AST.make<AssignStmt>();
    // Tier 3 — multi-return functions need one LHS per output for
    // the static -emit-* pipeline to refine ALL output types to
    // f64 (a single-LHS call keeps the trailing returns at `none`).
    // Use `[_p1, _p2, ..., _pK]` for stateful subsystems with K
    // returns; a single `__mflowlink_priming` keeps the stateless
    // case readable.
    if (Fn->Outputs.size() > 1) {
      for (size_t I = 0; I < Fn->Outputs.size(); ++I) {
        Driver->LHS.push_back(B.name("__mflowlink_priming" +
                                       std::to_string(I + 1)));
      }
    } else {
      Driver->LHS.push_back(B.name("__mflowlink_priming"));
    }
    Driver->RHS = Call;
    Driver->Suppressed = true;
    auto *S = AST.make<Script>();
    S->Body = AST.make<Block>();
    S->Body->Stmts.push_back(Driver);
    TU->ScriptNode = S;
  }
  return TU;
}

//===----------------------------------------------------------------------===//
// Tier 2 — subsystem metadata + per-target class-wrapper rendering.
//
// `describeSubsystem` recomputes the public surface (inputs, outputs,
// state slots) without touching the AST.  `emitSubsystemClassWrapper`
// renders a small target-specific class/struct that bundles the
// functional `step(...)` into the more ergonomic "class with mutating
// step(u) → y" idiom — matches the user's pick during planning (see
// docs/embedded_coder_roadmap.md §5).
//===----------------------------------------------------------------------===//

std::optional<SubsystemMeta> describeSubsystem(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::DiagnosticEngine &Diag) {
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (!Sub) {
    Diag.error(SourceLocation{}, "subsystem \"" + SubsystemName +
                                     "\" not found in `.mflow` file");
    return std::nullopt;
  }
  SubsystemMeta M;
  M.Name = sanitizeIdent(SubsystemName);

  // Inputs.
  auto Inports = collectPorts(*Sub, "signal_inport");
  for (auto &P : Inports) M.InputNames.push_back(P.Var);

  // Outputs.
  auto Outports = collectPorts(*Sub, "signal_outport");
  for (auto &P : Outports) M.OutputNames.push_back(P.Var);

  // Stateful blocks → one state slot each. Capture the per-block
  // initial condition so the class wrapper can default-init each
  // member field to the right value (matches the simulator's
  // t = 0 snapshot).
  for (const auto &N : Sub->Nodes) {
    if (!isStatefulKind(N.Kind)) continue;
    M.StateArgNames.push_back("s_" + sanitizeIdent(N.Id));
    M.StateReturnNames.push_back("s_" + sanitizeIdent(N.Id) + "_next");
    M.StateInitVals.push_back(initialStateOf(N));
  }
  return M;
}

namespace {

// Convert "snake_case" → "SnakeCase" so we get an idiomatic class
// name (Python: `DiscretePid`, C++: `DiscretePid`, …).
std::string toCamelCase(const std::string &S) {
  std::string Out;
  bool Up = true;
  for (char C : S) {
    if (C == '_' || C == '-') { Up = true; continue; }
    Out.push_back(Up ? std::toupper((unsigned char)C) : C);
    Up = false;
  }
  return Out;
}

std::string joinComma(const std::vector<std::string> &V) {
  std::string Out;
  for (size_t I = 0; I < V.size(); ++I) {
    if (I) Out += ", ";
    Out += V[I];
  }
  return Out;
}

} // namespace

std::string emitSubsystemClassWrapper(const SubsystemMeta &M,
                                       const std::string &Target) {
  std::string ClassName = toCamelCase(M.Name);
  std::ostringstream OS;
  bool Stateful = !M.StateArgNames.empty();
  size_t N = M.InputNames.size();   // public arg count
  size_t Outs = M.OutputNames.size(); // public return count

  // Tier 4 — render a state slot's default-init expression. Falls
  // back to 0.0 when the .mflow didn't capture an initial value
  // (early lowering paths may produce a SubsystemMeta without IC
  // info — older code paths).
  auto initFor = [&](size_t I) -> std::string {
    if (I >= M.StateInitVals.size()) return "0.0";
    std::ostringstream Tmp;
    Tmp.precision(17);
    Tmp << M.StateInitVals[I];
    return Tmp.str();
  };

  if (Target == "python") {
    OS << "\n\n";
    OS << "# Tier-2 class wrapper: object-style step() that carries\n";
    OS << "# the per-block state across calls.  Mirrors the functional\n";
    OS << "# `" << M.Name << "(...)` above; idiomatic for service / ML\n";
    OS << "# code that owns one controller instance per signal.\n";
    OS << "class " << ClassName << ":\n";
    OS << "    def __init__(self):\n";
    if (!Stateful) {
      OS << "        pass\n";
    } else {
      for (size_t I = 0; I < M.StateArgNames.size(); ++I)
        OS << "        self." << M.StateArgNames[I] << " = "
           << initFor(I) << "\n";
    }
    OS << "    def step(self";
    for (auto &In : M.InputNames) OS << ", " << In;
    OS << "):\n";
    OS << "        ";
    // LHS — y outputs + state-next.
    std::vector<std::string> Lhs;
    for (auto &Y : M.OutputNames) Lhs.push_back(Y);
    for (auto &S : M.StateReturnNames) Lhs.push_back(S);
    if (Lhs.size() == 1) OS << Lhs[0];
    else OS << joinComma(Lhs);
    OS << " = " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", self." << S;
    OS << ")\n";
    // Latch state.
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "        self." << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << "\n";
    }
    if (Outs == 1) OS << "        return " << M.OutputNames[0] << "\n";
    else           OS << "        return " << joinComma(M.OutputNames)
                      << "\n";
    return OS.str();
  }

  if (Target == "cpp") {
    OS << "\n\n";
    OS << "// Tier-2 class wrapper: object-style step() that carries\n";
    OS << "// the per-block state across calls.\n";
    OS << "struct " << ClassName << " {\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I)
      OS << "  double " << M.StateArgNames[I] << " = "
         << initFor(I) << ";\n";
    if (Outs == 1) OS << "  double step(";
    else           OS << "  std::tuple<";
    if (Outs > 1) {
      for (size_t I = 0; I < Outs; ++I) {
        if (I) OS << ", ";
        OS << "double";
      }
      OS << "> step(";
    }
    for (size_t I = 0; I < N; ++I) {
      if (I) OS << ", ";
      OS << "double " << M.InputNames[I];
    }
    OS << ") {\n";
    if (Stateful || Outs > 1) {
      OS << "    ";
      OS << "auto _r = " << M.Name << "(";
      OS << joinComma(M.InputNames);
      for (auto &S : M.StateArgNames) OS << ", " << S;
      OS << ");\n";
      // Latch state: tuple elements after the y outputs.
      for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
        OS << "    " << M.StateArgNames[I] << " = std::get<"
           << (Outs + I) << ">(_r);\n";
      }
      if (Outs == 1) {
        OS << "    return std::get<0>(_r);\n";
      } else {
        // Pack the y outputs into a tuple to return.
        OS << "    return std::make_tuple(";
        for (size_t I = 0; I < Outs; ++I) {
          if (I) OS << ", ";
          OS << "std::get<" << I << ">(_r)";
        }
        OS << ");\n";
      }
    } else {
      // Stateless single-output — direct return.
      OS << "    return " << M.Name << "(" << joinComma(M.InputNames)
         << ");\n";
    }
    OS << "  }\n";
    OS << "};\n";
    return OS.str();
  }

  if (Target == "c") {
    // C: struct + free `<ClassName>_step` taking pointer-to-self.
    OS << "\n\n";
    OS << "// Tier-2 class wrapper (C): struct + free step function.\n";
    OS << "typedef struct {\n";
    if (M.StateArgNames.empty()) {
      OS << "  char _unused;  /* zero-state placeholder */\n";
    } else {
      for (auto &S : M.StateArgNames) OS << "  double " << S << ";\n";
    }
    OS << "} " << ClassName << ";\n";
    // Tier 4 — init helper that lays down the initial conditions
    // captured from the .mflow. Callers do `<Class> obj; <Class>_init(&obj);`
    // before the first step.
    if (Stateful) {
      OS << "static void " << ClassName << "_init(" << ClassName
         << " *self) {\n";
      for (size_t I = 0; I < M.StateArgNames.size(); ++I)
        OS << "  self->" << M.StateArgNames[I] << " = " << initFor(I)
           << ";\n";
      OS << "}\n";
    }
    if (Outs == 1) OS << "static double ";
    else           OS << "static void ";
    OS << ClassName << "_step(" << ClassName << " *self";
    for (auto &In : M.InputNames) OS << ", double " << In;
    if (Outs > 1) {
      for (auto &Y : M.OutputNames) OS << ", double *" << Y << "_out";
    }
    OS << ") {\n";
    // Allocate locals for the function's full set of returns.
    std::vector<std::string> AllOuts;
    for (auto &Y : M.OutputNames) AllOuts.push_back("y_" + Y);
    for (auto &S : M.StateReturnNames) AllOuts.push_back(S);
    for (auto &L : AllOuts) OS << "  double " << L << ";\n";
    OS << "  " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", self->" << S;
    for (auto &L : AllOuts) OS << ", &" << L;
    OS << ");\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "  self->" << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << ";\n";
    }
    if (Outs == 1) {
      OS << "  return y_" << M.OutputNames[0] << ";\n";
    } else {
      for (auto &Y : M.OutputNames) {
        OS << "  *" << Y << "_out = y_" << Y << ";\n";
      }
    }
    OS << "}\n";
    return OS.str();
  }

  if (Target == "typescript") {
    OS << "\n\n";
    OS << "// Tier-2 class wrapper: object-style step() that carries\n";
    OS << "// the per-block state across calls.\n";
    OS << "class " << ClassName << " {\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I)
      OS << "  " << M.StateArgNames[I] << ": number = "
         << initFor(I) << ";\n";
    OS << "  step(";
    for (size_t I = 0; I < N; ++I) {
      if (I) OS << ", ";
      OS << M.InputNames[I] << ": number";
    }
    OS << "): ";
    if (Outs == 1) OS << "number";
    else {
      OS << "[";
      for (size_t I = 0; I < Outs; ++I) {
        if (I) OS << ", ";
        OS << "number";
      }
      OS << "]";
    }
    OS << " {\n";
    // Destructure the call result.
    OS << "    const [";
    std::vector<std::string> AllReturns;
    for (auto &Y : M.OutputNames) AllReturns.push_back(Y);
    for (auto &S : M.StateReturnNames) AllReturns.push_back(S);
    OS << joinComma(AllReturns);
    OS << "] = " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", this." << S;
    OS << ");\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "    this." << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << ";\n";
    }
    if (Outs == 1) {
      OS << "    return " << M.OutputNames[0] << ";\n";
    } else {
      OS << "    return [";
      OS << joinComma(M.OutputNames);
      OS << "];\n";
    }
    OS << "  }\n";
    OS << "}\n";
    return OS.str();
  }

  // Unsupported target — return empty so the caller skips the
  // wrapper. SystemVerilog handles state via registers natively
  // (Tier 5) and doesn't go through this wrapper.
  return {};
}

} // namespace matlab::flowchart
