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

#include <algorithm>
#include <cctype>
#include <map>
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

  for (const auto &E : F.Edges) {
    if (!InternalSet.count(E.To.Node)) continue;
    if (!InternalSet.count(E.From.Node)) continue; // inport / sink-edge
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

AssignStmt *lowerBlock(const Node &N, const std::string &OutVar,
                       const std::vector<Expr *> &Ins,
                       const std::vector<std::string> &InPortIds,
                       ASTBuilder &B, DiagnosticEngine &Diag) {
  const auto &K = N.Kind;

  auto get = [&](size_t I) -> Expr * {
    return I < Ins.size() ? Ins[I] : static_cast<Expr *>(B.number(0.0));
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
      return B.assign(OutVar, B.number(D));
    }
    // Strip the brackets, parse row-major numbers, build a MatrixLiteral.
    auto L = V.find('['), R = V.rfind(']');
    std::string Inner = V.substr(L + 1, R - L - 1);
    auto *ML = B.Ctx.make<MatrixLiteral>();
    std::string Tok;
    std::vector<Expr *> Row;
    auto endTok = [&]() {
      if (Tok.empty()) return;
      try { Row.push_back(B.number(std::stod(Tok))); }
      catch (...) { Row.push_back(B.number(0.0)); }
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
    auto *G = B.number(Gain);
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
    if (!Acc) Acc = B.number(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_product") {
    Expr *Acc = nullptr;
    for (Expr *T : Ins) {
      Acc = Acc ? B.bin(BinOp::ElemMul, Acc, T) : T;
    }
    if (!Acc) Acc = B.number(1.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_abs") {
    return B.assign(OutVar, B.call("abs", {get(0)}));
  }
  if (K == "signal_saturation") {
    double Lo = paramD(N, "lowerLimit", -1.0);
    double Hi = paramD(N, "upperLimit",  1.0);
    // The natural form `max(Lo, min(Hi, u))` routes through the
    // polymorphic matlab_min / matlab_max runtime entries — those
    // pick the *matrix* dispatch when any operand is none-typed
    // (function-arg-typed slots are `none` until refined), and the
    // emit-python pass can't lower the resulting !llvm.ptr return.
    // Emit the equivalent pure-arith form instead:
    //
    //   y = u + (Hi - u) * (u > Hi) + (Lo - u) * (u < Lo)
    //
    // Both correction terms are zero in the middle, exactly one is
    // active outside the rails — stays scalar f64 throughout.
    auto *U = get(0);
    // (Hi - u) * (u > Hi)
    auto *DHi  = B.bin(BinOp::Sub, B.number(Hi), U);
    auto *GtHi = B.bin(BinOp::Gt,  U, B.number(Hi));
    auto *CHi  = B.bin(BinOp::ElemMul, DHi, GtHi);
    // (Lo - u) * (u < Lo)
    auto *DLo  = B.bin(BinOp::Sub, B.number(Lo), U);
    auto *LtLo = B.bin(BinOp::Lt,  U, B.number(Lo));
    auto *CLo  = B.bin(BinOp::ElemMul, DLo, LtLo);
    // u + CHi + CLo
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
    if (Args.empty()) Args.push_back(B.number(0.0));
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
    if (!Acc) Acc = B.number(0.0);
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
    return B.assign(OutVar, B.bin(BO, get(0), B.number(0.0)));
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
    return B.assign(OutVar, B.bin(BO, get(0), B.number(C)));
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
    auto *Gt = B.bin(BinOp::Gt, get(1), B.number(Th));
    auto *Le = B.bin(BinOp::Le, get(1), B.number(Th));
    auto *T  = B.bin(BinOp::ElemMul, Gt, get(0));
    auto *F  = B.bin(BinOp::ElemMul, Le, get(2));
    auto *Sum = B.bin(BinOp::Add, T, F);
    auto *Anchored = B.bin(BinOp::Add, B.number(0.0), Sum);
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
    if (!Acc) Acc = B.number(0.0);
    return B.assign(OutVar, Acc);
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
    matlab::DiagnosticEngine &Diag) {
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

  for (auto *N : Internal) {
    VarOfNode[N->Id] = uniqueVarFor(N->Id);
  }

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

  // Emit one statement per internal block.
  for (auto *N : Internal) {
    auto Ports = inputPortsOf(*N);
    std::vector<Expr *> Ins;
    for (auto &P : Ports) Ins.push_back(resolveInputExpr(N->Id, P));
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

  // Build the function node.
  auto *Fn = AST.make<Function>();
  Fn->Name = AST.intern(sanitizeIdent(SubsystemName));
  for (auto &P : Inports)  Fn->Inputs.push_back(AST.intern(P.Var));
  for (auto &P : Outports) Fn->Outputs.push_back(AST.intern(P.Var));
  Fn->Body = Body;
  return Fn;
}

matlab::TranslationUnit *buildSubsystemTU(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag) {
  auto *Fn = lowerSubsystemToMatlab(Doc, SubsystemName, AST, Diag);
  if (!Fn) return nullptr;

  auto *TU = AST.make<TranslationUnit>();
  TU->Functions.push_back(Fn);

  // Synthesise a driver script that calls the function with concrete
  // f64 args. That call site forces the static `-emit-*` pipeline to
  // refine the function's slots to `double` instead of `none` / `void*`
  // (the function-only-file pitfall documented during the §17.5 #8
  // work). The driver is a single AssignStmt placed in the TU's
  // Script node — matlab_llvm allows a script body before function
  // definitions in the same file.
  ASTBuilder B{AST};
  if (!Fn->Inputs.empty()) {
    std::vector<Expr *> Args;
    for (size_t I = 0; I < Fn->Inputs.size(); ++I)
      Args.push_back(B.number(0.0));
    auto *Call = B.call(std::string(Fn->Name), std::move(Args));
    auto *Driver = AST.make<AssignStmt>();
    auto *Lhs = B.name("__mflowlink_priming");
    Driver->LHS.push_back(Lhs);
    Driver->RHS = Call;
    Driver->Suppressed = true;
    auto *S = AST.make<Script>();
    S->Body = AST.make<Block>();
    S->Body->Stmts.push_back(Driver);
    TU->ScriptNode = S;
  }
  return TU;
}

} // namespace matlab::flowchart
