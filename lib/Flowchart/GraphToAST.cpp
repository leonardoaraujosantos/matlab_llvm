#include "matlab/Flowchart/GraphToAST.h"

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"

#include <fstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace matlab::flowchart {

namespace {

//===----------------------------------------------------------------------===//
// Phase 3 reducer: structured control flow over flowchart graphs.
//
// Strategy: a recursive walker that builds AST blocks one statement at a
// time, with two helpers for branching shapes:
//
//   1. `findJoin(T, F)` — given the two successors of an `if` node,
//      walks both branches with `stepOver` (which jumps over any
//      already-visited nested control structure) and returns the first
//      node both branches reach. This is the if's reconvergence point.
//
//   2. Stop sets — when we recurse into a sub-region (a then/else/body
//      block), we add the region's exit node id(s) to a per-call Stop
//      set. The walker stops on entry to any node in Stop, leaving
//      outer code to pick up the continuation. This is what gives
//      structured nesting without a global dominator analysis: each
//      level only knows about its own boundary.
//
// Loops use the same mechanism — the loop head's id becomes a Stop
// boundary while walking the body, so the body's back-edge cleanly
// terminates the body walk without us ever revisiting the loop head.
//
// Trade-offs vs. a full dominator/post-dominator pass:
//   - Cheaper to implement and debug; matches the way the IDE will
//     emit diagrams (single reconvergence, single back-edge).
//   - Refuses irreducible CFGs cleanly (findJoin returns "", which
//     surfaces as a "branches don't reconverge" diagnostic).
//   - Will need to be replaced if the schema later allows multi-entry
//     loops, computed gotos, or fused regions. None of those are on
//     the roadmap for v1.
//===----------------------------------------------------------------------===//

// TU-level state shared across every Builder so custom-block-provided
// functions are inserted at most once and named-flow lifts don't clash
// with library functions.
struct TUContext {
  const BuildOptions &Opts;
  TranslationUnit &TU;
  std::set<std::string> InsertedFunctions; // by MATLAB function name
  // Phase 8a: optional output map from .mflow (file_id, line) to the
  // originating block id. Populated as each block is emitted so the
  // DAP / LSP servers can surface block context on debug stops.
  // nullptr when the caller doesn't need it.
  BlockLineMap *BlockMap = nullptr;
};

class Builder {
public:
  Builder(const Flow &F, const FlowDoc &Doc, ASTContext &Ctx,
          SourceManager &SM, DiagnosticEngine &Diag, TUContext &TUC)
      : F(F), Doc(Doc), Ctx(Ctx), SM(SM), Diag(Diag), TUC(TUC) {
    for (auto &N : F.Nodes) NodeById[N.Id] = &N;
    for (auto &E : F.Edges)
      if (E.Kind == "control") OutEdges[E.From.Node].push_back(&E);
  }

  bool emit(Block &Out) {
    const Node *Start = nullptr;
    for (auto &N : F.Nodes)
      if (N.Kind == "start") { Start = &N; break; }
    if (!Start) {
      Diag.error(F.Loc, "flow \"" + F.Name + "\" has no start node");
      return false;
    }
    auto &Outs = OutEdges[Start->Id];
    if (Outs.empty()) {
      Diag.error(Start->Loc, "start node has no outgoing control edge");
      return false;
    }
    if (Outs.size() != 1) {
      Diag.error(Start->Loc,
                 "start node must have exactly one outgoing edge, found " +
                     std::to_string(Outs.size()));
      return false;
    }
    walk(Outs[0]->To.Node, /*Stop=*/{}, Out.Stmts);
    return !Diag.hasErrors();
  }

private:
  const Flow &F;
  const FlowDoc &Doc;
  ASTContext &Ctx;
  SourceManager &SM;
  DiagnosticEngine &Diag;
  TUContext &TUC;
  std::unordered_map<std::string, const Node *> NodeById;
  std::unordered_map<std::string, std::vector<const Edge *>> OutEdges;
  std::unordered_map<std::string, std::string> IfJoinCache;

  const Node *node(const std::string &Id) {
    auto It = NodeById.find(Id);
    return It == NodeById.end() ? nullptr : It->second;
  }

  // Single-successor advance. Reports a diagnostic if the node has 0
  // or 2+ outgoing control edges.
  std::string singleOut(const Node &N) {
    auto It = OutEdges.find(N.Id);
    if (It == OutEdges.end() || It->second.empty()) return "";
    if (It->second.size() != 1) {
      Diag.error(N.Loc,
                 "node \"" + N.Id + "\" (kind \"" + N.Kind +
                     "\") has multiple outgoing control edges; only "
                     "if/for/while may branch");
      return "";
    }
    return It->second[0]->To.Node;
  }

  // Phase 8a: record this block's `(file_id, line)` → block id in the
  // optional output map. Called from every site that emits a block —
  // the linear walk, handleIf / handleFor / handleWhile, and the
  // custom-block call site. The DAP server reads the map back when
  // formatting stack frames.
  void recordBlock(const Node &N) {
    if (!TUC.BlockMap) return;
    if (!N.Loc.isValid()) return;
    auto LC = SM.getLineColumn(N.Loc);
    if (LC.Line == 0) return;
    TUC.BlockMap->Lookup[BlockLineMap::key(N.Loc.File, LC.Line)] = N.Id;
  }

  std::string portOut(const Node &N, const std::string &PortId) {
    auto It = OutEdges.find(N.Id);
    if (It != OutEdges.end()) {
      for (auto *E : It->second)
        if (E->From.Port == PortId) return E->To.Node;
    }
    Diag.error(N.Loc, "node \"" + N.Id + "\" (kind \"" + N.Kind +
                          "\") has no outgoing edge on port \"" + PortId +
                          "\"");
    return "";
  }

  //===--------------------------------------------------------------===//
  // Reconvergence detection
  //===--------------------------------------------------------------===//

  // Symbolic "next reachable exit" of `Id`: where does linear control
  // continue from this node, treating nested control structures as
  // single super-nodes? Used by findJoin's two-pointer walk.
  std::string stepOver(const std::string &Id) {
    const Node *N = node(Id);
    if (!N) return "";
    if (N->Kind == "end" || N->Kind == "break" ||
        N->Kind == "continue" || N->Kind == "return") {
      return "";
    }
    if (N->Kind == "if") {
      auto It = IfJoinCache.find(Id);
      if (It != IfJoinCache.end()) return It->second;
      std::string T, Fa;
      auto OutsIt = OutEdges.find(Id);
      if (OutsIt != OutEdges.end()) {
        for (auto *E : OutsIt->second) {
          if (E->From.Port == "true") T = E->To.Node;
          else if (E->From.Port == "false") Fa = E->To.Node;
        }
      }
      // Tentatively cache empty to break recursion through a self-
      // referential pathology before computing the real value.
      IfJoinCache[Id] = "";
      std::string J = findJoin(T, Fa, /*OuterStop=*/{});
      IfJoinCache[Id] = J;
      return J;
    }
    if (N->Kind == "for" || N->Kind == "while") {
      auto OutsIt = OutEdges.find(Id);
      if (OutsIt != OutEdges.end()) {
        for (auto *E : OutsIt->second)
          if (E->From.Port == "done") return E->To.Node;
      }
      return "";
    }
    if (N->Kind == "switch") {
      // Phase 8e: stepOver for a switch jumps to the multi-branch
      // join. Compute it the same way handleSwitch does so the
      // findJoin two-pointer walk treats the switch as a single
      // super-node. Cheap to recompute (no memoisation needed —
      // the inputs are the node's port targets, all known here).
      std::vector<std::string> AllTargets;
      auto OutsIt = OutEdges.find(Id);
      if (OutsIt != OutEdges.end()) {
        for (auto *E : OutsIt->second) AllTargets.push_back(E->To.Node);
      }
      if (AllTargets.empty()) return "";
      std::string J = AllTargets[0];
      for (size_t I = 1; I < AllTargets.size() && !J.empty(); ++I) {
        J = findJoin(J, AllTargets[I], /*OuterStop=*/{});
      }
      return J;
    }
    if (N->Kind == "try") {
      // Phase 8e: stepOver for a try jumps to the body/catch join.
      std::string Body, Catch;
      auto OutsIt = OutEdges.find(Id);
      if (OutsIt != OutEdges.end()) {
        for (auto *E : OutsIt->second) {
          if (E->From.Port == "body") Body = E->To.Node;
          else if (E->From.Port == "catch") Catch = E->To.Node;
        }
      }
      return findJoin(Body, Catch, /*OuterStop=*/{});
    }
    // Linear: follow the single out edge. Don't fail loudly here —
    // findJoin tolerates dead-ends.
    auto OutsIt = OutEdges.find(Id);
    if (OutsIt == OutEdges.end() || OutsIt->second.empty()) return "";
    if (OutsIt->second.size() == 1) return OutsIt->second[0]->To.Node;
    return ""; // multi-out non-branching node — let walk() surface this
  }

  std::string findJoin(const std::string &T, const std::string &Fa,
                       const std::unordered_set<std::string> &OuterStop) {
    // Trivial cases first: if both branches start at the same node OR
    // one branch starts at the other's start (degenerate if-no-else),
    // that node is the join.
    if (T.empty() && Fa.empty()) return "";
    if (T == Fa) return T;

    std::unordered_set<std::string> SeenT, SeenF;
    if (!T.empty()) SeenT.insert(T);
    if (!Fa.empty()) SeenF.insert(Fa);
    if (!Fa.empty() && SeenT.count(Fa)) return Fa;
    if (!T.empty() && SeenF.count(T)) return T;

    std::string CurT = T, CurF = Fa;
    // Hard cap to defend against pathological graphs that don't
    // converge — the cap is the total node count plus slack.
    size_t Steps = 0, Limit = NodeById.size() * 4 + 8;
    while ((!CurT.empty() || !CurF.empty()) && ++Steps <= Limit) {
      if (!CurT.empty()) {
        if (OuterStop.count(CurT)) {
          CurT = "";
        } else {
          CurT = stepOver(CurT);
          if (!CurT.empty()) {
            if (SeenF.count(CurT)) return CurT;
            SeenT.insert(CurT);
          }
        }
      }
      if (!CurF.empty()) {
        if (OuterStop.count(CurF)) {
          CurF = "";
        } else {
          CurF = stepOver(CurF);
          if (!CurF.empty()) {
            if (SeenT.count(CurF)) return CurF;
            SeenF.insert(CurF);
          }
        }
      }
    }
    return "";
  }

  //===--------------------------------------------------------------===//
  // Per-block source synthesis (linear blocks)
  //===--------------------------------------------------------------===//

  static std::string quoteSingle(const std::string &S) {
    std::string Q;
    Q.reserve(S.size() + 2);
    Q += '\'';
    for (char C : S) Q += (C == '\'') ? "''" : std::string(1, C);
    Q += '\'';
    return Q;
  }

  // Render a non-control-flow block to MATLAB source (a single
  // statement terminated by ';\n'). Empty result means the block
  // contributes no statement (start/end/comment).
  std::string renderLinearSource(const Node &N) {
    auto require = [&](std::string_view Key) -> const std::string * {
      auto *V = N.getData(Key);
      if (!V || V->empty()) {
        Diag.error(N.Loc, "block kind \"" + N.Kind + "\" (id \"" + N.Id +
                              "\") requires non-empty data field \"" +
                              std::string(Key) + "\"");
        return nullptr;
      }
      return V;
    };
    auto getOr = [&](std::string_view Key, const std::string &Default) {
      auto *V = N.getData(Key);
      return (V && !V->empty()) ? *V : Default;
    };

    if (N.Kind == "start" || N.Kind == "end" || N.Kind == "comment")
      return "";

    if (N.Kind == "variable" || N.Kind == "constant") {
      auto *Name = require("name");
      auto *Val  = require("value");
      if (!Name || !Val) return "";
      return *Name + " = " + *Val + ";\n";
    }
    if (N.Kind == "assignment") {
      auto *L = require("lhs");
      auto *R = require("rhs");
      if (!L || !R) return "";
      return *L + " = " + *R + ";\n";
    }
    if (N.Kind == "expression") {
      auto *E = require("expression");
      if (!E) return "";
      std::string S = *E;
      if (S.empty() || S.back() != ';') S += ';';
      return S + "\n";
    }
    if (N.Kind == "display") {
      auto *E = require("expression");
      if (!E) return "";
      return "disp(" + *E + ");\n";
    }
    if (N.Kind == "input") {
      auto *Name = require("name");
      if (!Name) return "";
      const std::string Prompt = getOr("prompt", "");
      if (Prompt.empty()) return *Name + " = input('');\n";
      return *Name + " = input(" + quoteSingle(Prompt) + ");\n";
    }
    if (N.Kind == "function_call") {
      auto *Callee = require("callee");
      if (!Callee) return "";
      const std::string Args = getOr("args", "");
      const std::string Lhs  = getOr("lhs",  "");
      std::string Call = *Callee + "(" + Args + ")";
      return Lhs.empty() ? (Call + ";\n") : (Lhs + " = " + Call + ";\n");
    }
    if (N.Kind == "matrix_literal") {
      auto *Name = require("name");
      auto *Rows = require("rows");
      if (!Name || !Rows) return "";
      return *Name + " = [" + *Rows + "];\n";
    }

    // `function_definition` is a visual marker that the program uses a
    // function defined as a separate flow; the function body is lifted
    // out at the FlowDoc level (see `buildAST`). The block itself
    // contributes no statement, but we validate the cross-reference
    // here so a bad `flow_id` fails at AST-build time, not later.
    if (N.Kind == "function_definition") {
      auto *FlowId = N.getData("flow_id");
      if (FlowId && !FlowId->empty()) {
        bool Found = false;
        for (auto &Other : Doc.Flows) {
          if (Other.Id == *FlowId) { Found = true; break; }
        }
        if (!Found) {
          Diag.error(N.Loc, "function_definition block (id \"" + N.Id +
                                "\") references unknown flow id \"" +
                                *FlowId + "\"");
        }
      }
      return "";
    }

    // `subflow_call` is the typed counterpart of `function_call`: it
    // names the target by `flow_id` instead of by callee name. We
    // resolve the flow's name and emit the same `lhs = name(args);`
    // shape — the lifted Function carries the matching signature.
    if (N.Kind == "subflow_call") {
      auto *FlowId = require("flow_id");
      if (!FlowId) return "";
      const Flow *Target = nullptr;
      for (auto &Other : Doc.Flows) {
        if (Other.Id == *FlowId) { Target = &Other; break; }
      }
      if (!Target) {
        Diag.error(N.Loc, "subflow_call block references unknown flow id \"" +
                              *FlowId + "\"");
        return "";
      }
      const std::string Args = getOr("args", "");
      const std::string Lhs  = getOr("lhs",  "");
      std::string Call = Target->Name + "(" + Args + ")";
      return Lhs.empty() ? (Call + ";\n") : (Lhs + " = " + Call + ";\n");
    }

    Diag.error(N.Loc, "unknown block kind \"" + N.Kind + "\"");
    return "";
  }

  std::vector<Stmt *> parseSynthesized(const std::string &Src,
                                       const std::string &BufferName) {
    if (Src.empty()) return {};
    FileID FF = SM.addBuffer(BufferName, Src);
    Lexer Lx(SM, FF, Diag);
    auto Toks = Lx.tokenize();
    if (Diag.hasErrors()) return {};
    Parser P(std::move(Toks), Ctx, Diag);
    TranslationUnit *Mini = P.parseFile();
    if (!Mini || !Mini->ScriptNode || !Mini->ScriptNode->Body) return {};
    return Mini->ScriptNode->Body->Stmts;
  }

  // Synthesize "EXPR;\n", parse it, and return the inner Expr. The
  // helper exists so `if`'s `data.cond` and `for`'s `data.iter` can
  // ride the existing Pratt parser without duplicating expression
  // grammar inside this layer.
  Expr *parseExprFromString(const std::string &Str,
                            const std::string &BufferName,
                            SourceLocation BlockLoc) {
    std::string Src = Str + ";\n";
    auto Stmts = parseSynthesized(Src, BufferName);
    if (Diag.hasErrors()) return nullptr;
    if (Stmts.size() != 1 || Stmts[0]->Kind != NodeKind::ExprStmt) {
      Diag.error(BlockLoc, "expected a single expression in block field, "
                           "got " + std::to_string(Stmts.size()) +
                           " statement(s)");
      return nullptr;
    }
    return static_cast<ExprStmt *>(Stmts[0])->E;
  }

  //===--------------------------------------------------------------===//
  // Walker
  //===--------------------------------------------------------------===//

  // Walk control flow from `Start`, appending statements to `Out`.
  // Returns the id of the node we stopped on (empty if we reached
  // `end` or ran out of edges). The walk halts when:
  //   - Cur is in Stop (= region boundary)
  //   - Cur is null / not found
  //   - Cur is an `end` node
  //   - We emit a `break` / `continue` / `return` (dead code follows)
  std::string walk(const std::string &Start,
                   const std::unordered_set<std::string> &Stop,
                   std::vector<Stmt *> &Out) {
    std::string Cur = Start;
    while (!Cur.empty() && !Stop.count(Cur) && !Diag.hasErrors()) {
      const Node *N = node(Cur);
      if (!N) return "";
      if (N->Kind == "end") return Cur;
      if (N->Kind == "start") {
        Cur = singleOut(*N);
        continue;
      }

      if (N->Kind == "if") {
        Cur = handleIf(*N, Stop, Out);
        continue;
      }
      if (N->Kind == "for") {
        Cur = handleFor(*N, Stop, Out);
        continue;
      }
      if (N->Kind == "while") {
        Cur = handleWhile(*N, Stop, Out);
        continue;
      }
      if (N->Kind == "break") {
        auto *S = Ctx.make<BreakStmt>();
        S->Range.Begin = N->Loc;
        Out.push_back(S);
        return "";
      }
      if (N->Kind == "continue") {
        auto *S = Ctx.make<ContinueStmt>();
        S->Range.Begin = N->Loc;
        Out.push_back(S);
        return "";
      }
      if (N->Kind == "return") {
        auto *S = Ctx.make<ReturnStmt>();
        S->Range.Begin = N->Loc;
        Out.push_back(S);
        return "";
      }
      if (N->Kind == "switch") {
        Cur = handleSwitch(*N, Stop, Out);
        continue;
      }
      if (N->Kind == "try") {
        Cur = handleTry(*N, Stop, Out);
        continue;
      }
      if (N->Kind == "custom") {
        Cur = handleCustom(*N, Out);
        continue;
      }

      // Linear block: render → parse → append, then advance.
      std::string Src = renderLinearSource(*N);
      if (Diag.hasErrors()) return "";
      auto Stmts = parseSynthesized(Src, "<flow:" + N->Id + ">");
      if (Diag.hasErrors()) return "";
      recordBlock(*N);
      for (auto *S : Stmts) {
        // Phase 6: remap to the originating block's location in the
        // .mflow source so DAP breakpoints set on a block's JSON
        // line fire when execution reaches the synthesized
        // statements. The synthetic per-block buffer keeps the
        // parser's intra-field locations available for diagnostics
        // that point inside `data.expression` text — only the
        // outer Stmt range is overridden here.
        S->Range.Begin = N->Loc;
        if (!S->Range.End.isValid()) S->Range.End = N->Loc;
        Out.push_back(S);
      }

      Cur = singleOut(*N);
    }
    return Cur;
  }

  std::string handleIf(const Node &N,
                       const std::unordered_set<std::string> &OuterStop,
                       std::vector<Stmt *> &Out) {
    auto *CondStr = N.getData("cond");
    if (!CondStr || CondStr->empty()) {
      Diag.error(N.Loc, "block kind \"if\" (id \"" + N.Id +
                            "\") requires non-empty data field \"cond\"");
      return "";
    }
    std::string T  = portOut(N, "true");
    std::string Fa = portOut(N, "false");
    if (Diag.hasErrors()) return "";

    std::string J = findJoin(T, Fa, OuterStop);

    Expr *Cond = parseExprFromString(*CondStr, "<flow:" + N.Id + ":cond>",
                                     N.Loc);
    if (!Cond) return "";

    recordBlock(N);
    auto *S = Ctx.make<IfStmt>();
    S->Range.Begin = N.Loc;
    S->Cond = Cond;
    S->Then = Ctx.make<Block>();
    S->Else = Ctx.make<Block>();

    auto InnerStop = OuterStop;
    if (!J.empty()) InnerStop.insert(J);

    walk(T,  InnerStop, S->Then->Stmts);
    if (Diag.hasErrors()) return "";
    walk(Fa, InnerStop, S->Else->Stmts);
    if (Diag.hasErrors()) return "";

    // If the false branch went directly to the join with no statements
    // along the way, the user drew an if-without-else. Drop the empty
    // Else block so the formatter produces `if ... end` instead of
    // `if ... else end`. (We keep the IfStmt itself even when both
    // branches are empty — that's a valid no-op the diagram contains.)
    if (S->Else->Stmts.empty()) S->Else = nullptr;

    Out.push_back(S);
    return J;
  }

  std::string handleFor(const Node &N,
                        const std::unordered_set<std::string> &OuterStop,
                        std::vector<Stmt *> &Out) {
    auto *VarStr  = N.getData("var");
    auto *IterStr = N.getData("iter");
    if (!VarStr || VarStr->empty()) {
      Diag.error(N.Loc, "block kind \"for\" (id \"" + N.Id +
                            "\") requires non-empty data field \"var\"");
      return "";
    }
    if (!IterStr || IterStr->empty()) {
      Diag.error(N.Loc, "block kind \"for\" (id \"" + N.Id +
                            "\") requires non-empty data field \"iter\"");
      return "";
    }
    std::string Body = portOut(N, "body");
    std::string Done = portOut(N, "done");
    if (Diag.hasErrors()) return "";

    Expr *Iter = parseExprFromString(*IterStr, "<flow:" + N.Id + ":iter>",
                                     N.Loc);
    if (!Iter) return "";

    recordBlock(N);
    auto *S = Ctx.make<ForStmt>();
    S->Range.Begin = N.Loc;
    S->Var = Ctx.intern(*VarStr);
    S->Iter = Iter;
    S->Body = Ctx.make<Block>();

    // Body terminates when control hits a back-edge to this loop head.
    auto InnerStop = OuterStop;
    InnerStop.insert(N.Id);

    walk(Body, InnerStop, S->Body->Stmts);
    if (Diag.hasErrors()) return "";

    Out.push_back(S);
    return Done;
  }

  //===--------------------------------------------------------------===//
  // Custom blocks (Phase 4b): user-supplied MATLAB attached to a block.
  //
  // A `custom` block has three provenance modes for its function body:
  //   - `data.source`     — inline MATLAB text in the JSON
  //   - `data.path`       — `.m` file relative to the .mflow's directory
  //   - `data.library_id` — name resolved against the block search path
  // Exactly one must be set. The block also carries call-site fields
  // (`callee`, `args`, `lhs`) that mirror `function_call`. We insert
  // the resolved Function into TU.Functions at most once per name —
  // many custom blocks pointing at the same library function share the
  // same AST node.
  //===--------------------------------------------------------------===//

  static std::string dirOf(const std::string &Path) {
    auto Slash = Path.find_last_of("/\\");
    return Slash == std::string::npos ? std::string() : Path.substr(0, Slash);
  }

  static bool isAbsolute(const std::string &P) {
    return !P.empty() && (P.front() == '/' || P.front() == '\\');
  }

  static bool fileExists(const std::string &P) {
    std::ifstream T(P);
    return T.good();
  }

  // Resolve a library_id like "matforge.dsp/fir_4tap" against the
  // search path. Each search entry is tried in order; the first hit
  // wins. Trailing `.m` is appended automatically.
  std::string resolveLibraryId(const std::string &LibId) {
    if (LibId.empty()) return "";
    std::string FileName = LibId;
    if (FileName.size() < 2 ||
        FileName.substr(FileName.size() - 2) != ".m") {
      FileName += ".m";
    }
    for (const auto &Dir : TUC.Opts.BlockSearchPath) {
      std::string Candidate = Dir + "/" + FileName;
      if (fileExists(Candidate)) return Candidate;
    }
    return "";
  }

  std::string handleCustom(const Node &N, std::vector<Stmt *> &Out) {
    auto get = [&](std::string_view Key) -> const std::string * {
      return N.getData(Key);
    };

    const std::string *Source = get("source");
    const std::string *Path   = get("path");
    const std::string *LibId  = get("library_id");
    int Provenance = (Source && !Source->empty() ? 1 : 0) +
                     (Path && !Path->empty() ? 1 : 0) +
                     (LibId && !LibId->empty() ? 1 : 0);
    if (Provenance == 0) {
      Diag.error(N.Loc, "custom block (id \"" + N.Id + "\") must specify "
                            "exactly one of \"source\", \"path\", or "
                            "\"library_id\"");
      return "";
    }
    if (Provenance > 1) {
      Diag.error(N.Loc, "custom block (id \"" + N.Id + "\") sets multiple "
                            "of \"source\" / \"path\" / \"library_id\"; "
                            "pick one");
      return "";
    }

    auto *Callee = N.getData("callee");
    auto *NameField = N.getData("name");
    if ((!Callee || Callee->empty()) && (!NameField || NameField->empty())) {
      Diag.error(N.Loc, "custom block (id \"" + N.Id + "\") requires "
                            "either \"callee\" or \"name\" to identify "
                            "the function to insert");
      return "";
    }
    const std::string &CalleeName =
        (Callee && !Callee->empty()) ? *Callee : *NameField;

    // Insert the function body once per name. Subsequent custom
    // blocks pointing at the same function (e.g. a shared library
    // primitive) reuse the already-inserted AST.
    if (TUC.InsertedFunctions.count(CalleeName) == 0) {
      if (!insertCustomFunction(N, Source, Path, LibId, CalleeName)) return "";
      TUC.InsertedFunctions.insert(CalleeName);
    }

    auto *ArgsP = N.getData("args");
    auto *LhsP  = N.getData("lhs");
    const std::string Args = ArgsP ? *ArgsP : "";
    const std::string Lhs  = LhsP  ? *LhsP  : "";

    std::string Call = CalleeName + "(" + Args + ")";
    std::string Src = Lhs.empty() ? (Call + ";\n")
                                  : (Lhs + " = " + Call + ";\n");
    auto Stmts = parseSynthesized(Src, "<flow:" + N.Id + ":call>");
    if (Diag.hasErrors()) return "";
    recordBlock(N);
    for (auto *S : Stmts) {
      // Phase 6: see remap rationale above the linear-block path.
      S->Range.Begin = N.Loc;
      if (!S->Range.End.isValid()) S->Range.End = N.Loc;
      Out.push_back(S);
    }

    return singleOut(N);
  }

  // Pull the function body into the SourceManager (preserving the
  // original file path for `path` / `library_id` so DAP step-into
  // and LSP go-to-definition land on real disk locations), parse it,
  // pick out the named Function, and append to TU.Functions.
  bool insertCustomFunction(const Node &N, const std::string *Source,
                            const std::string *Path,
                            const std::string *LibId,
                            const std::string &Name) {
    FileID FF = 0;
    std::string ResolvedPath;

    if (Source && !Source->empty()) {
      FF = SM.addBuffer("<flow:" + N.Id + ":source>", *Source);
    } else if (Path && !Path->empty()) {
      ResolvedPath = isAbsolute(*Path) ? *Path
                                       : (TUC.Opts.MflowDirectory.empty()
                                              ? *Path
                                              : TUC.Opts.MflowDirectory +
                                                    "/" + *Path);
      FF = SM.loadFile(ResolvedPath);
      if (!FF) {
        Diag.error(N.Loc, "custom block (id \"" + N.Id +
                              "\") cannot open path: " + ResolvedPath);
        return false;
      }
    } else if (LibId && !LibId->empty()) {
      ResolvedPath = resolveLibraryId(*LibId);
      if (ResolvedPath.empty()) {
        std::string PathDesc;
        for (size_t I = 0; I < TUC.Opts.BlockSearchPath.size(); ++I) {
          if (I) PathDesc += ":";
          PathDesc += TUC.Opts.BlockSearchPath[I];
        }
        if (PathDesc.empty()) PathDesc = "(empty — set MATFORGE_BLOCK_PATH "
                                          "or pass --block-path)";
        Diag.error(N.Loc, "custom block (id \"" + N.Id +
                              "\") could not resolve library_id \"" +
                              *LibId + "\" against block search path " +
                              PathDesc);
        return false;
      }
      FF = SM.loadFile(ResolvedPath);
      if (!FF) {
        Diag.error(N.Loc, "custom block (id \"" + N.Id +
                              "\") cannot open library file: " +
                              ResolvedPath);
        return false;
      }
    }

    Lexer Lx(SM, FF, Diag);
    auto Toks = Lx.tokenize();
    if (Diag.hasErrors()) return false;
    Parser P(std::move(Toks), Ctx, Diag);
    TranslationUnit *Mini = P.parseFile();
    if (!Mini || Diag.hasErrors()) return false;

    Function *Found = nullptr;
    for (Function *Fn : Mini->Functions) {
      if (Fn && Fn->Name == Name) { Found = Fn; break; }
    }
    if (!Found) {
      Diag.error(N.Loc, "custom block (id \"" + N.Id +
                            "\") body does not define a function named \"" +
                            Name + "\"");
      return false;
    }

    // Optional arity check. When the IDE supplies `inputs` /
    // `outputs` arrays, fail fast if they disagree with the parsed
    // signature — much clearer than a downstream "wrong number of
    // arguments" diagnostic from Sema.
    if (auto *Ins = N.getDataArray("inputs")) {
      if (Ins->size() != Found->Inputs.size()) {
        Diag.error(N.Loc, "custom block (id \"" + N.Id + "\") declares " +
                              std::to_string(Ins->size()) + " input(s) but "
                              "function \"" + Name + "\" takes " +
                              std::to_string(Found->Inputs.size()));
        return false;
      }
    }
    if (auto *Outs = N.getDataArray("outputs")) {
      if (Outs->size() != Found->Outputs.size()) {
        Diag.error(N.Loc, "custom block (id \"" + N.Id + "\") declares " +
                              std::to_string(Outs->size()) + " output(s) "
                              "but function \"" + Name + "\" returns " +
                              std::to_string(Found->Outputs.size()));
        return false;
      }
    }

    TUC.TU.Functions.push_back(Found);
    return true;
  }

  // Phase 8e: switch reduction. The block's `data.discriminant`
  // names the switch expression and `data.cases` (string array)
  // names one expression per case in port order. Ports are
  // `case_0`..`case_<N-1>` plus `default`. Each case's chain
  // reconverges at a single join (the next statement after the
  // switch). The join is computed by pairwise-reducing all
  // outgoing branch heads through the existing two-pointer
  // `findJoin` — iteratively folding the running result against
  // each next branch keeps the implementation small at the cost
  // of O(N) extra walks for an N-case switch (acceptable: real
  // switches rarely exceed a handful of cases).
  std::string handleSwitch(const Node &N,
                           const std::unordered_set<std::string> &OuterStop,
                           std::vector<Stmt *> &Out) {
    auto *DiscStr = N.getData("discriminant");
    if (!DiscStr || DiscStr->empty()) {
      Diag.error(N.Loc,
                 "block kind \"switch\" (id \"" + N.Id +
                     "\") requires non-empty data field \"discriminant\"");
      return "";
    }
    auto *CaseValues = N.getDataArray("cases");
    if (!CaseValues || CaseValues->empty()) {
      Diag.error(N.Loc,
                 "block kind \"switch\" (id \"" + N.Id +
                     "\") requires non-empty data array \"cases\"");
      return "";
    }

    // Collect each case branch's first node id from the matching
    // port. `case_<i>` for the i-th case, plus `default`.
    std::vector<std::string> CaseTargets;
    for (size_t I = 0; I < CaseValues->size(); ++I) {
      std::string Port = "case_" + std::to_string(I);
      std::string T = portOut(N, Port);
      if (Diag.hasErrors()) return "";
      CaseTargets.push_back(T);
    }
    std::string DefaultTarget = portOut(N, "default");
    if (Diag.hasErrors()) return "";

    // Multi-branch join via iterated pairwise reduction.
    std::vector<std::string> AllTargets = CaseTargets;
    AllTargets.push_back(DefaultTarget);
    std::string J;
    if (!AllTargets.empty()) {
      J = AllTargets[0];
      for (size_t I = 1; I < AllTargets.size(); ++I) {
        J = findJoin(J, AllTargets[I], OuterStop);
        if (J.empty()) break;
      }
    }

    Expr *Disc = parseExprFromString(*DiscStr,
                                     "<flow:" + N.Id + ":discriminant>",
                                     N.Loc);
    if (!Disc) return "";

    recordBlock(N);
    auto *S = Ctx.make<SwitchStmt>();
    S->Range.Begin = N.Loc;
    S->Discriminant = Disc;

    auto InnerStop = OuterStop;
    if (!J.empty()) InnerStop.insert(J);

    for (size_t I = 0; I < CaseValues->size(); ++I) {
      Expr *V = parseExprFromString((*CaseValues)[I],
                                    "<flow:" + N.Id + ":case" +
                                        std::to_string(I) + ">",
                                    N.Loc);
      if (!V) return "";
      SwitchCase SC;
      SC.Value = V;
      SC.Body = Ctx.make<Block>();
      walk(CaseTargets[I], InnerStop, SC.Body->Stmts);
      if (Diag.hasErrors()) return "";
      S->Cases.push_back(std::move(SC));
    }

    // `default` lowers to a Case with Value = nullptr (MATLAB's
    // `otherwise` clause).
    {
      SwitchCase SC;
      SC.Value = nullptr;
      SC.Body = Ctx.make<Block>();
      walk(DefaultTarget, InnerStop, SC.Body->Stmts);
      if (Diag.hasErrors()) return "";
      S->Cases.push_back(std::move(SC));
    }

    Out.push_back(S);
    return J;
  }

  // Phase 8e: try / catch reduction. The block has `body` and
  // `catch` ports; both branches reconverge after the try.
  // `data.catch_var` (optional) is the exception variable name
  // in the catch body's scope.
  std::string handleTry(const Node &N,
                        const std::unordered_set<std::string> &OuterStop,
                        std::vector<Stmt *> &Out) {
    std::string BodyTarget = portOut(N, "body");
    std::string CatchTarget = portOut(N, "catch");
    if (Diag.hasErrors()) return "";

    std::string J = findJoin(BodyTarget, CatchTarget, OuterStop);

    recordBlock(N);
    auto *S = Ctx.make<TryStmt>();
    S->Range.Begin = N.Loc;
    if (auto *CV = N.getData("catch_var"); CV && !CV->empty())
      S->CatchVar = Ctx.intern(*CV);
    S->TryBody = Ctx.make<Block>();
    S->CatchBody = Ctx.make<Block>();

    auto InnerStop = OuterStop;
    if (!J.empty()) InnerStop.insert(J);

    walk(BodyTarget, InnerStop, S->TryBody->Stmts);
    if (Diag.hasErrors()) return "";
    walk(CatchTarget, InnerStop, S->CatchBody->Stmts);
    if (Diag.hasErrors()) return "";

    Out.push_back(S);
    return J;
  }

  std::string handleWhile(const Node &N,
                          const std::unordered_set<std::string> &OuterStop,
                          std::vector<Stmt *> &Out) {
    auto *CondStr = N.getData("cond");
    if (!CondStr || CondStr->empty()) {
      Diag.error(N.Loc, "block kind \"while\" (id \"" + N.Id +
                            "\") requires non-empty data field \"cond\"");
      return "";
    }
    std::string Body = portOut(N, "body");
    std::string Done = portOut(N, "done");
    if (Diag.hasErrors()) return "";

    Expr *Cond = parseExprFromString(*CondStr, "<flow:" + N.Id + ":cond>",
                                     N.Loc);
    if (!Cond) return "";

    recordBlock(N);
    auto *S = Ctx.make<WhileStmt>();
    S->Range.Begin = N.Loc;
    S->Cond = Cond;
    S->Body = Ctx.make<Block>();

    auto InnerStop = OuterStop;
    InnerStop.insert(N.Id);

    walk(Body, InnerStop, S->Body->Stmts);
    if (Diag.hasErrors()) return "";

    Out.push_back(S);
    return Done;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

// Lift one `function`-kind flow to a top-level Function AST. The body
// uses the same Builder as the entry flow, so every Phase 2/3 control-
// flow shape (if / for / while / break / ...) — and Phase 4b custom
// blocks — works inside functions just like it does in the program.
static Function *liftFunctionFlow(const Flow &F, const FlowDoc &Doc,
                                  ASTContext &Ctx, SourceManager &SM,
                                  DiagnosticEngine &Diag, TUContext &TUC) {
  auto *Fn = Ctx.make<Function>();
  Fn->Range.Begin = F.Loc;
  Fn->Name = Ctx.intern(F.Name);
  for (auto &In : F.Sig.Inputs)
    Fn->Inputs.push_back(Ctx.intern(In));
  for (auto &Out : F.Sig.Outputs)
    Fn->Outputs.push_back(Ctx.intern(Out));
  Fn->Body = Ctx.make<Block>();

  Builder B(F, Doc, Ctx, SM, Diag, TUC);
  if (!B.emit(*Fn->Body)) return nullptr;
  return Fn;
}

TranslationUnit *buildAST(const FlowDoc &Doc, ASTContext &Ctx,
                          SourceManager &SM, DiagnosticEngine &Diag,
                          const BuildOptions &Opts,
                          BlockLineMap *OutBlockMap) {
  const Flow *Entry = Doc.entryFlow();
  if (!Entry) {
    SourceLocation Invalid;
    Diag.error(Invalid, "flowchart has no entry flow named \"" + Doc.Entry +
                            "\"");
    return nullptr;
  }
  if (Entry->Kind != "program") {
    Diag.error(Entry->Loc, "entry flow \"" + Entry->Name +
                               "\" must have kind \"program\", not \"" +
                               Entry->Kind + "\"");
    return nullptr;
  }

  auto *TU = Ctx.make<TranslationUnit>();
  TUContext TUC{Opts, *TU, /*InsertedFunctions=*/{},
                /*BlockMap=*/OutBlockMap};

  // Lift every non-entry function-kind flow before the program walk.
  // Names must be unique across the TU (the textual MATLAB frontend
  // has the same constraint — siblings sharing a name shadow each
  // other), so guard against accidental duplicates.
  for (auto &Other : Doc.Flows) {
    if (&Other == Entry) continue;
    if (Other.Kind != "function") continue;
    if (!TUC.InsertedFunctions.insert(Other.Name).second) {
      Diag.error(Other.Loc, "duplicate function name \"" + Other.Name +
                                "\" across flows");
      return nullptr;
    }
    Function *Fn = liftFunctionFlow(Other, Doc, Ctx, SM, Diag, TUC);
    if (!Fn) return nullptr;
    TU->Functions.push_back(Fn);
  }

  auto *S = Ctx.make<Script>();
  S->Body = Ctx.make<Block>();
  TU->ScriptNode = S;

  Builder B(*Entry, Doc, Ctx, SM, Diag, TUC);
  if (!B.emit(*S->Body)) return nullptr;
  return TU;
}

} // namespace matlab::flowchart
