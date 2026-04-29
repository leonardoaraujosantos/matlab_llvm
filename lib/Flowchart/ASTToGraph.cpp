#include "matlab/Flowchart/ASTToGraph.h"

#include "matlab/AST/AST.h"
#include "matlab/AST/Formatter.h"

#include <algorithm>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace matlab::flowchart {

namespace {

//===----------------------------------------------------------------------===//
// IR for the emitter — mirror the loader's shape so future code can
// share the structures, but we don't depend on the loader headers here.
//===----------------------------------------------------------------------===//

struct OutPort {
  std::string Id;
};

struct OutEndpoint {
  std::string Node;
  std::string Port;
};

struct OutEdge {
  std::string Id;
  std::string Kind = "control";
  OutEndpoint From;
  OutEndpoint To;
};

struct OutData {
  // Order matters for IDE-byte-identical output? No — JSON object
  // keys are emitted in alphabetical order regardless of insertion.
  // Use std::map for that ordering automatically.
  std::map<std::string, std::string> Scalar;
  std::map<std::string, std::vector<std::string>> Arrays;
};

struct OutNode {
  std::string Id;
  std::string Kind;
  std::string Label;
  OutData Data;
  std::vector<OutPort> InPorts;
  std::vector<OutPort> OutPorts;
  int X = 0, Y = 0;
};

struct OutFlow {
  std::string Id;
  std::string Kind;
  std::string Name;
  std::vector<std::string> SigInputs;
  std::vector<std::string> SigOutputs;
  std::vector<OutNode> Nodes;
  std::vector<OutEdge> Edges;
};

struct OutDoc {
  std::string Entry;
  std::vector<OutFlow> Flows;
};

//===----------------------------------------------------------------------===//
// Wiring abstraction
//
// A `Pad` is the set of (node_id, port_id) endpoints that a future
// block's `in` port should be wired from. Linear stmts produce a
// single-source pad ({last_block, "out"}). If/else branches produce
// a multi-source pad (then-tail's out + else-tail's out). Break /
// continue / return produce an empty pad — control terminates.
//===----------------------------------------------------------------------===//

struct Pad {
  std::vector<std::pair<std::string, std::string>> Sources;
  bool empty() const { return Sources.empty(); }
};

class FlowBuilder {
public:
  explicit FlowBuilder(OutFlow &F) : F(F) {}

  // Allocate a fresh node with stable id `n_<kind>_<n>` per kind.
  OutNode &addNode(const std::string &Kind) {
    auto &Idx = KindCount[Kind];
    ++Idx;
    OutNode N;
    N.Id = "n_" + Kind + "_" + std::to_string(Idx);
    N.Kind = Kind;
    F.Nodes.push_back(std::move(N));
    return F.Nodes.back();
  }

  // Some structural blocks want fixed ids (`s` for the sole start,
  // `e` for the sole end of a flow). Use this to allocate them.
  OutNode &addNamed(const std::string &Id, const std::string &Kind) {
    OutNode N;
    N.Id = Id;
    N.Kind = Kind;
    F.Nodes.push_back(std::move(N));
    return F.Nodes.back();
  }

  void addEdge(const std::string &FromN, const std::string &FromP,
               const std::string &ToN, const std::string &ToP) {
    OutEdge E;
    E.Id = "e_" + std::to_string(++EdgeCounter);
    E.From = {FromN, FromP};
    E.To = {ToN, ToP};
    F.Edges.push_back(std::move(E));
  }

  // Wire every source in `From` to (`ToN`, `ToP`).
  void wire(const Pad &From, const std::string &ToN,
            const std::string &ToP) {
    for (auto &[N, P] : From.Sources) addEdge(N, P, ToN, ToP);
  }

private:
  OutFlow &F;
  std::map<std::string, int> KindCount;
  int EdgeCounter = 0;
};

//===----------------------------------------------------------------------===//
// Expression / statement → MATLAB text helpers
//===----------------------------------------------------------------------===//

std::string formatExprStr(const Expr *E) {
  if (!E) return "";
  std::ostringstream OS;
  formatExpr(OS, *E);
  return OS.str();
}

// Render a comma-separated argument list — e.g. for `function_call`
// blocks' `data.args`, where MATLAB's `f(a, b, c)` shows up as a
// `CallOrIndex` whose `Args` we want to print without the outer
// `f(...)` wrapper.
std::string formatCallArgs(const std::vector<Expr *> &Args) {
  std::string Out;
  for (size_t i = 0; i < Args.size(); ++i) {
    if (i) Out += ", ";
    Out += formatExprStr(Args[i]);
  }
  return Out;
}

// MatrixLiteral's interior (rows separated by `; `, elements within
// a row separated by spaces). Used for `matrix_literal` blocks'
// `data.rows` field.
std::string formatMatrixRows(const MatrixLiteral &M) {
  std::string Out;
  for (size_t r = 0; r < M.Rows.size(); ++r) {
    if (r) Out += "; ";
    auto &Row = M.Rows[r];
    for (size_t c = 0; c < Row.size(); ++c) {
      if (c) Out += " ";
      Out += formatExprStr(Row[c]);
    }
  }
  return Out;
}

bool isCallOf(const Expr *E, std::string_view Name) {
  if (!E || E->Kind != NodeKind::CallOrIndex) return false;
  auto &C = static_cast<const CallOrIndex &>(*E);
  if (!C.Callee || C.Callee->Kind != NodeKind::NameExpr) return false;
  return static_cast<const NameExpr &>(*C.Callee).Name == Name;
}

bool isLiteralValue(const Expr *E) {
  if (!E) return false;
  switch (E->Kind) {
  case NodeKind::IntegerLiteral:
  case NodeKind::FPLiteral:
  case NodeKind::ImagLiteral:
  case NodeKind::StringLiteral:
  case NodeKind::CharLiteral:
    return true;
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(*E);
    return (U.Op == UnOp::Plus || U.Op == UnOp::Minus) &&
           isLiteralValue(U.Operand);
  }
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Stmt walker — emits one block (or a control-structure subgraph)
// and returns the exit Pad.
//===----------------------------------------------------------------------===//

class StmtWalker {
public:
  StmtWalker(FlowBuilder &FB) : FB(FB) {}

  Pad walkBlock(const Block &B, Pad In) {
    for (Stmt *S : B.Stmts) {
      if (!S) continue;
      In = walkStmt(*S, In);
      if (In.empty()) break; // hit a terminator (break/continue/return)
    }
    return In;
  }

  Pad walkStmt(const Stmt &S, Pad In) {
    switch (S.Kind) {
    case NodeKind::AssignStmt:
      return walkAssign(static_cast<const AssignStmt &>(S), In);
    case NodeKind::ExprStmt:
      return walkExprStmt(static_cast<const ExprStmt &>(S), In);
    case NodeKind::IfStmt:
      return walkIf(static_cast<const IfStmt &>(S), In);
    case NodeKind::ForStmt:
      return walkFor(static_cast<const ForStmt &>(S), In);
    case NodeKind::WhileStmt:
      return walkWhile(static_cast<const WhileStmt &>(S), In);
    case NodeKind::BreakStmt:
      return walkSimple(In, "break", {});
    case NodeKind::ContinueStmt:
      return walkSimple(In, "continue", {});
    case NodeKind::ReturnStmt:
      return walkSimple(In, "return", {});
    case NodeKind::Block:
      return walkBlock(static_cast<const Block &>(S), In);
    default:
      // Switch / try / global / persistent / etc. — degrade to an
      // expression block carrying the formatted source. Round-trip
      // is lossy here; the IDE rendering will show the raw text.
      return emitExpression(formatStmtFallback(S), In);
    }
  }

private:
  FlowBuilder &FB;

  // Empty Pad means "no successor" — used by terminators.
  Pad noContinuation() { return Pad{}; }

  Pad pad1(const std::string &N, const std::string &P) {
    Pad Out;
    Out.Sources.emplace_back(N, P);
    return Out;
  }

  Pad mergePads(const Pad &A, const Pad &B) {
    Pad Out = A;
    for (auto &S : B.Sources) Out.Sources.push_back(S);
    return Out;
  }

  static OutPort port(std::string Id) { return OutPort{std::move(Id)}; }

  // Allocate a "linear" node with one in / one out port. Most blocks
  // are this shape; if/for/while customize ports below.
  OutNode &linearNode(const std::string &Kind) {
    auto &N = FB.addNode(Kind);
    N.InPorts.push_back(port("in"));
    N.OutPorts.push_back(port("out"));
    return N;
  }

  Pad walkAssign(const AssignStmt &A, Pad In) {
    if (A.LHS.size() != 1 || A.RHS == nullptr) {
      // Multi-LHS like `[u, v] = f(x)` — degrade to expression.
      return emitExpression(formatStmtFallback(A), In);
    }
    const Expr *L = A.LHS[0];
    // `name = literal-or-signed-literal` → `variable` block.
    if (L && L->Kind == NodeKind::NameExpr && isLiteralValue(A.RHS)) {
      auto &NE = static_cast<const NameExpr &>(*L);
      auto &N = linearNode("variable");
      N.Data.Scalar["name"] = std::string(NE.Name);
      N.Data.Scalar["value"] = formatExprStr(A.RHS);
      FB.wire(In, N.Id, "in");
      return pad1(N.Id, "out");
    }
    // `name = MATRIX_LITERAL` → `matrix_literal` block.
    if (L && L->Kind == NodeKind::NameExpr &&
        A.RHS->Kind == NodeKind::MatrixLiteral) {
      auto &NE = static_cast<const NameExpr &>(*L);
      auto &M = static_cast<const MatrixLiteral &>(*A.RHS);
      auto &N = linearNode("matrix_literal");
      N.Data.Scalar["name"] = std::string(NE.Name);
      N.Data.Scalar["rows"] = formatMatrixRows(M);
      FB.wire(In, N.Id, "in");
      return pad1(N.Id, "out");
    }
    // `lhs = call(args)` where lhs is a plain name → `function_call`
    // with `lhs`. (`subflow_call` is name-vs-id, indistinguishable
    // from the AST — emit `function_call`.)
    if (L && L->Kind == NodeKind::NameExpr &&
        A.RHS->Kind == NodeKind::CallOrIndex) {
      auto &Call = static_cast<const CallOrIndex &>(*A.RHS);
      if (Call.Callee && Call.Callee->Kind == NodeKind::NameExpr) {
        auto &NE = static_cast<const NameExpr &>(*L);
        auto &CN = static_cast<const NameExpr &>(*Call.Callee);
        auto &N = linearNode("function_call");
        N.Data.Scalar["callee"] = std::string(CN.Name);
        N.Data.Scalar["args"] = formatCallArgs(Call.Args);
        N.Data.Scalar["lhs"] = std::string(NE.Name);
        FB.wire(In, N.Id, "in");
        return pad1(N.Id, "out");
      }
    }
    // General assignment — `assignment` block with formatted lhs / rhs.
    auto &N = linearNode("assignment");
    N.Data.Scalar["lhs"] = formatExprStr(L);
    N.Data.Scalar["rhs"] = formatExprStr(A.RHS);
    FB.wire(In, N.Id, "in");
    return pad1(N.Id, "out");
  }

  Pad walkExprStmt(const ExprStmt &S, Pad In) {
    const Expr *E = S.E;
    if (!E) return In;
    // `disp(EXPR)` → `display` block.
    if (E->Kind == NodeKind::CallOrIndex) {
      auto &Call = static_cast<const CallOrIndex &>(*E);
      if (isCallOf(E, "disp") && Call.Args.size() == 1) {
        auto &N = linearNode("display");
        N.Data.Scalar["expression"] = formatExprStr(Call.Args[0]);
        FB.wire(In, N.Id, "in");
        return pad1(N.Id, "out");
      }
      // Generic `name(args)` standalone call → `function_call` block.
      if (Call.Callee && Call.Callee->Kind == NodeKind::NameExpr) {
        auto &CN = static_cast<const NameExpr &>(*Call.Callee);
        auto &N = linearNode("function_call");
        N.Data.Scalar["callee"] = std::string(CN.Name);
        N.Data.Scalar["args"] = formatCallArgs(Call.Args);
        FB.wire(In, N.Id, "in");
        return pad1(N.Id, "out");
      }
    }
    // Free-form expression (compound calls, struct access, etc.) —
    // emit as `expression` block carrying the formatted text.
    return emitExpression(formatExprStr(E), In);
  }

  // Reference returned by addNode is invalidated as soon as the
  // vector grows on a subsequent addNode call. Any walker that
  // recurses into a body (and therefore adds more nodes) MUST snapshot
  // the id as a std::string before the recursion. The helpers below
  // capture the id immediately, fill the new node's fields, and then
  // use only the local copy across the body walk.

  Pad walkIf(const IfStmt &I, Pad In) {
    std::string Id;
    {
      auto &N = FB.addNode("if");
      N.InPorts.push_back(port("in"));
      N.OutPorts.push_back(port("true"));
      N.OutPorts.push_back(port("false"));
      N.Data.Scalar["cond"] = formatExprStr(I.Cond);
      Id = N.Id;
    }
    FB.wire(In, Id, "in");

    // Then branch.
    Pad ThenExit = pad1(Id, "true");
    if (I.Then) ThenExit = walkBlock(*I.Then, ThenExit);

    // The current AST shape supports IfStmt::Elseifs. Re-fold them
    // into nested IfStmt-shaped emission on the false branch so the
    // .mflow surface stays simple (no `elseif` block kind needed).
    Pad ElseExit = pad1(Id, "false");
    if (!I.Elseifs.empty()) {
      // Build a chained IfStmt for the elseifs + final else.
      ElseExit = walkElseifChain(I.Elseifs, 0, I.Else, ElseExit);
    } else if (I.Else) {
      ElseExit = walkBlock(*I.Else, ElseExit);
    }

    return mergePads(ThenExit, ElseExit);
  }

  // Walk an elseif chain starting at index `i` plus the trailing
  // optional `else`, all wired from the `false` branch of the parent
  // if (or the previous chain link).
  Pad walkElseifChain(const std::vector<ElseIf> &Eis, size_t i,
                      const Block *FinalElse, Pad In) {
    if (i >= Eis.size()) {
      if (FinalElse) return walkBlock(*FinalElse, In);
      return In; // nothing on this branch — flow through
    }
    const ElseIf &E = Eis[i];
    std::string Id;
    {
      auto &N = FB.addNode("if");
      N.InPorts.push_back(port("in"));
      N.OutPorts.push_back(port("true"));
      N.OutPorts.push_back(port("false"));
      N.Data.Scalar["cond"] = formatExprStr(E.Cond);
      Id = N.Id;
    }
    FB.wire(In, Id, "in");
    Pad ThenExit = pad1(Id, "true");
    if (E.Body) ThenExit = walkBlock(*E.Body, ThenExit);
    Pad ElseExit = walkElseifChain(Eis, i + 1, FinalElse, pad1(Id, "false"));
    return mergePads(ThenExit, ElseExit);
  }

  Pad walkFor(const ForStmt &F, Pad In) {
    std::string Id;
    {
      auto &N = FB.addNode("for");
      N.InPorts.push_back(port("in"));
      N.OutPorts.push_back(port("body"));
      N.OutPorts.push_back(port("done"));
      N.Data.Scalar["var"] = std::string(F.Var);
      N.Data.Scalar["iter"] = formatExprStr(F.Iter);
      Id = N.Id;
    }
    FB.wire(In, Id, "in");

    Pad BodyExit = pad1(Id, "body");
    if (F.Body) BodyExit = walkBlock(*F.Body, BodyExit);
    // Back-edge: body tail re-enters the loop head's `in` port.
    if (!BodyExit.empty()) FB.wire(BodyExit, Id, "in");

    return pad1(Id, "done");
  }

  Pad walkWhile(const WhileStmt &W, Pad In) {
    std::string Id;
    {
      auto &N = FB.addNode("while");
      N.InPorts.push_back(port("in"));
      N.OutPorts.push_back(port("body"));
      N.OutPorts.push_back(port("done"));
      N.Data.Scalar["cond"] = formatExprStr(W.Cond);
      Id = N.Id;
    }
    FB.wire(In, Id, "in");

    Pad BodyExit = pad1(Id, "body");
    if (W.Body) BodyExit = walkBlock(*W.Body, BodyExit);
    if (!BodyExit.empty()) FB.wire(BodyExit, Id, "in");

    return pad1(Id, "done");
  }

  Pad walkSimple(Pad In, const std::string &Kind, std::vector<Expr *>) {
    auto &N = FB.addNode(Kind);
    N.InPorts.push_back(port("in"));
    N.OutPorts.push_back(port("out"));
    FB.wire(In, N.Id, "in");
    return noContinuation();
  }

  Pad emitExpression(const std::string &Text, Pad In) {
    auto &N = linearNode("expression");
    N.Data.Scalar["expression"] = Text;
    FB.wire(In, N.Id, "in");
    return pad1(N.Id, "out");
  }

  // Last-resort textual fallback for unhandled Stmt kinds.
  std::string formatStmtFallback(const Stmt &S) {
    // Wrap the stmt in a synthetic Script + TU and run formatAST,
    // then strip the trailing newline. Keeps us decoupled from
    // Formatter's internals at the cost of a small extra pass.
    Block B;
    B.Stmts = {const_cast<Stmt *>(&S)};
    Script Sc;
    Sc.Body = &B;
    TranslationUnit TU;
    TU.ScriptNode = &Sc;
    std::ostringstream OS;
    formatAST(OS, TU);
    std::string Out = OS.str();
    while (!Out.empty() && (Out.back() == '\n' || Out.back() == ';'))
      Out.pop_back();
    return Out;
  }
};

//===----------------------------------------------------------------------===//
// Auto-layout: assign x/y so the IDE has reasonable initial positions
// when it first loads the emitted document. Walks node insertion
// order; each node moves down 120 px in a single column. Branches
// don't get x-offset in v1 — they're laid out by the IDE on save.
//===----------------------------------------------------------------------===//

void layoutFlow(OutFlow &F) {
  int X = 200;
  int Y = 0;
  for (auto &N : F.Nodes) {
    N.X = X;
    N.Y = Y;
    Y += 120;
  }
}

//===----------------------------------------------------------------------===//
// JSON writer — IDE-format compatible (2-space indent, " : " around
// the colon, alphabetical keys, blank-line empty arrays/objects).
//===----------------------------------------------------------------------===//

class JsonWriter {
public:
  explicit JsonWriter(std::ostream &OS) : OS(OS) {}

  void writeDoc(const OutDoc &D) { writeRoot(D); OS << "\n"; }

private:
  std::ostream &OS;

  static std::string escape(std::string_view S) {
    std::string Out;
    Out.reserve(S.size() + 2);
    for (char C : S) {
      switch (C) {
      case '"':  Out += "\\\""; break;
      case '\\': Out += "\\\\"; break;
      case '\b': Out += "\\b";  break;
      case '\f': Out += "\\f";  break;
      case '\n': Out += "\\n";  break;
      case '\r': Out += "\\r";  break;
      case '\t': Out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(C) < 0x20) {
          char Buf[8];
          std::snprintf(Buf, sizeof Buf, "\\u%04x", (unsigned)C);
          Out += Buf;
        } else {
          Out += C;
        }
        break;
      }
    }
    return Out;
  }

  void indent(int D) { for (int i = 0; i < D; ++i) OS << "  "; }

  void emitString(std::string_view S) { OS << '"' << escape(S) << '"'; }

  // Emit a key/value pair; `Emit` is a callback that writes the value
  // with the given indent depth.
  template <typename F>
  void emitField(int D, std::string_view Key, F Emit) {
    indent(D);
    emitString(Key);
    OS << " : ";
    Emit(D);
  }

  void emitInt(int V) { OS << V; }

  // Empty arrays/objects use an IDE-style blank-line filler so files
  // round-trip through MatForge byte-identically.
  void emitEmptyArray(int D) {
    OS << "[\n\n";
    indent(D);
    OS << "]";
  }
  void emitEmptyObject(int D) {
    OS << "{\n\n";
    indent(D);
    OS << "}";
  }

  template <typename T, typename E>
  void emitArray(int D, const std::vector<T> &V, E EmitElem) {
    if (V.empty()) { emitEmptyArray(D); return; }
    OS << "[\n";
    for (size_t i = 0; i < V.size(); ++i) {
      indent(D + 1);
      EmitElem(D + 1, V[i]);
      if (i + 1 < V.size()) OS << ",";
      OS << "\n";
    }
    indent(D);
    OS << "]";
  }

  void emitStringArray(int D, const std::vector<std::string> &V) {
    emitArray(D, V, [&](int /*Dx*/, const std::string &S) {
      emitString(S);
    });
  }

  // OutData is sorted alphabetically across BOTH scalar and array
  // entries to match the IDE's flat alphabetical key order.
  void emitData(int D, const OutData &Data) {
    std::vector<std::pair<std::string, int>> Keys; // 0 = scalar, 1 = array
    Keys.reserve(Data.Scalar.size() + Data.Arrays.size());
    for (auto &P : Data.Scalar) Keys.emplace_back(P.first, 0);
    for (auto &P : Data.Arrays) Keys.emplace_back(P.first, 1);
    std::sort(Keys.begin(), Keys.end(),
              [](const auto &A, const auto &B) { return A.first < B.first; });
    if (Keys.empty()) { emitEmptyObject(D); return; }
    OS << "{\n";
    for (size_t i = 0; i < Keys.size(); ++i) {
      const auto &K = Keys[i].first;
      if (Keys[i].second == 0) {
        emitField(D + 1, K, [&](int) { emitString(Data.Scalar.at(K)); });
      } else {
        emitField(D + 1, K,
                  [&](int Dx) { emitStringArray(Dx, Data.Arrays.at(K)); });
      }
      if (i + 1 < Keys.size()) OS << ",";
      OS << "\n";
    }
    indent(D);
    OS << "}";
  }

  void emitPort(int D, const OutPort &P) {
    OS << "{\n";
    emitField(D + 1, "id", [&](int) { emitString(P.Id); });
    OS << "\n";
    indent(D);
    OS << "}";
  }

  void emitPortList(int D, const std::vector<OutPort> &Ps) {
    emitArray(D, Ps, [&](int Dx, const OutPort &P) { emitPort(Dx, P); });
  }

  void emitNode(int D, const OutNode &N) {
    OS << "{\n";
    emitField(D + 1, "data", [&](int Dx) { emitData(Dx, N.Data); });
    OS << ",\n";
    emitField(D + 1, "id", [&](int) { emitString(N.Id); });
    OS << ",\n";
    emitField(D + 1, "kind", [&](int) { emitString(N.Kind); });
    OS << ",\n";
    emitField(D + 1, "label", [&](int) { emitString(N.Label); });
    OS << ",\n";
    emitField(D + 1, "ports", [&](int Dx) {
      OS << "{\n";
      emitField(Dx + 1, "in", [&](int Dy) { emitPortList(Dy, N.InPorts); });
      OS << ",\n";
      emitField(Dx + 1, "out",
                [&](int Dy) { emitPortList(Dy, N.OutPorts); });
      OS << "\n";
      indent(Dx);
      OS << "}";
    });
    OS << ",\n";
    emitField(D + 1, "ui", [&](int Dx) {
      OS << "{\n";
      emitField(Dx + 1, "position", [&](int Dy) {
        OS << "{\n";
        indent(Dy + 1);
        OS << "\"x\" : " << N.X << ",\n";
        indent(Dy + 1);
        OS << "\"y\" : " << N.Y << "\n";
        indent(Dy);
        OS << "}";
      });
      OS << "\n";
      indent(Dx);
      OS << "}";
    });
    OS << "\n";
    indent(D);
    OS << "}";
  }

  void emitEndpoint(int D, const OutEndpoint &E) {
    OS << "{\n";
    emitField(D + 1, "node", [&](int) { emitString(E.Node); });
    OS << ",\n";
    emitField(D + 1, "port", [&](int) { emitString(E.Port); });
    OS << "\n";
    indent(D);
    OS << "}";
  }

  void emitEdge(int D, const OutEdge &E) {
    OS << "{\n";
    emitField(D + 1, "from",
              [&](int Dx) { emitEndpoint(Dx, E.From); });
    OS << ",\n";
    emitField(D + 1, "id", [&](int) { emitString(E.Id); });
    OS << ",\n";
    emitField(D + 1, "kind", [&](int) { emitString(E.Kind); });
    OS << ",\n";
    emitField(D + 1, "to", [&](int Dx) { emitEndpoint(Dx, E.To); });
    OS << "\n";
    indent(D);
    OS << "}";
  }

  void emitFlow(int D, const OutFlow &F) {
    OS << "{\n";
    emitField(D + 1, "edges", [&](int Dx) {
      emitArray(Dx, F.Edges,
                [&](int Dy, const OutEdge &E) { emitEdge(Dy, E); });
    });
    OS << ",\n";
    emitField(D + 1, "id", [&](int) { emitString(F.Id); });
    OS << ",\n";
    emitField(D + 1, "kind", [&](int) { emitString(F.Kind); });
    OS << ",\n";
    emitField(D + 1, "name", [&](int) { emitString(F.Name); });
    OS << ",\n";
    emitField(D + 1, "nodes", [&](int Dx) {
      emitArray(Dx, F.Nodes,
                [&](int Dy, const OutNode &N) { emitNode(Dy, N); });
    });
    OS << ",\n";
    emitField(D + 1, "signature", [&](int Dx) {
      OS << "{\n";
      emitField(Dx + 1, "inputs",
                [&](int Dy) { emitStringArray(Dy, F.SigInputs); });
      OS << ",\n";
      emitField(Dx + 1, "outputs",
                [&](int Dy) { emitStringArray(Dy, F.SigOutputs); });
      OS << "\n";
      indent(Dx);
      OS << "}";
    });
    OS << "\n";
    indent(D);
    OS << "}";
  }

  void writeRoot(const OutDoc &D) {
    OS << "{\n";
    emitField(1, "entry", [&](int) { emitString(D.Entry); });
    OS << ",\n";
    emitField(1, "flows", [&](int Dx) {
      emitArray(Dx, D.Flows,
                [&](int Dy, const OutFlow &F) { emitFlow(Dy, F); });
    });
    OS << ",\n";
    emitField(1, "schema", [&](int) {
      emitString("matforge.flowchart");
    });
    OS << ",\n";
    emitField(1, "settings", [&](int Dx) {
      OS << "{\n";
      indent(Dx + 1);
      OS << "\"columnMajor\" : true,\n";
      indent(Dx + 1);
      OS << "\"defaultNumericType\" : \"double\",\n";
      indent(Dx + 1);
      OS << "\"sourceLanguage\" : \"matforge\"\n";
      indent(Dx);
      OS << "}";
    });
    OS << ",\n";
    emitField(1, "version", [&](int) { emitString("0.1.0"); });
    OS << "\n}";
  }
};

//===----------------------------------------------------------------------===//
// Top-level orchestrator
//===----------------------------------------------------------------------===//

void emitFlowFromBlock(OutFlow &Out, const Block *Body) {
  FlowBuilder FB(Out);
  // Always-present structural anchors. The forward direction of the
  // pipeline (loadMflow + buildAST) requires exactly one start and at
  // least one end node per program flow; emit both deterministically.
  auto &Start = FB.addNamed("s", "start");
  Start.OutPorts.push_back(OutPort{"out"});
  auto &End = FB.addNamed("e", "end");
  End.InPorts.push_back(OutPort{"in"});

  Pad In;
  In.Sources.emplace_back("s", "out");

  StmtWalker SW(FB);
  Pad Exit = Body ? SW.walkBlock(*Body, In) : In;

  if (!Exit.empty()) FB.wire(Exit, "e", "in");

  // Move the `e` node to the bottom of the layout column. addNode'd
  // bodies got 0..(N-1)*120 from layoutFlow's later pass; we adjust
  // Start/End relative positions here.
  layoutFlow(Out);
  // Push the `e` node below all body nodes by re-placing it.
  for (auto &N : Out.Nodes) {
    if (N.Id == "e") N.Y += 60; // tiny gap below the last linear stmt
  }
}

OutFlow emitProgramFlow(const TranslationUnit &TU) {
  OutFlow F;
  F.Id = "flow_main";
  F.Kind = "program";
  F.Name = (TU.ScriptNode ? "main" : "main");
  emitFlowFromBlock(F, TU.ScriptNode ? TU.ScriptNode->Body : nullptr);
  return F;
}

OutFlow emitFunctionFlow(const Function &Fn) {
  OutFlow F;
  F.Id = "flow_" + std::string(Fn.Name);
  F.Kind = "function";
  F.Name = std::string(Fn.Name);
  for (auto &P : Fn.Inputs) F.SigInputs.emplace_back(P);
  for (auto &O : Fn.Outputs) F.SigOutputs.emplace_back(O);
  emitFlowFromBlock(F, Fn.Body);
  return F;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

void emitMflow(std::ostream &OS, const TranslationUnit &TU) {
  OutDoc D;
  D.Entry = "main";

  // Program flow first, then every top-level user function in
  // declaration order. Nested classes are not yet representable as
  // .mflow; their bodies fall back to expression blocks via the
  // formatStmtFallback path inside the walker.
  D.Flows.push_back(emitProgramFlow(TU));
  for (const Function *Fn : TU.Functions) {
    if (Fn) D.Flows.push_back(emitFunctionFlow(*Fn));
  }

  JsonWriter W(OS);
  W.writeDoc(D);
}

} // namespace matlab::flowchart
