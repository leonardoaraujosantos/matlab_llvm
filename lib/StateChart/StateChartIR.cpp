#include "matlab/StateChart/StateChartIR.h"

#include "matlab/Basic/Diagnostic.h"

#include <algorithm>
#include <cctype>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

namespace matlab::statechart {

const char *decompositionName(Decomposition D) {
  switch (D) {
  case Decomposition::Or:   return "or";
  case Decomposition::And:  return "and";
  case Decomposition::Leaf: return "leaf";
  }
  return "?";
}

const char *containerStyleName(ContainerStyle C) {
  switch (C) {
  case ContainerStyle::State:    return "state";
  case ContainerStyle::Subchart: return "subchart";
  case ContainerStyle::Box:      return "box";
  }
  return "?";
}

const char *junctionKindName(JunctionKind K) {
  switch (K) {
  case JunctionKind::Connective: return "connective";
  case JunctionKind::History:    return "history";
  case JunctionKind::Entry:      return "entry";
  case JunctionKind::Exit:       return "exit";
  case JunctionKind::Default:    return "default";
  }
  return "?";
}

const char *transitionKindName(TransitionKind K) {
  switch (K) {
  case TransitionKind::Outer:   return "outer";
  case TransitionKind::Inner:   return "inner";
  case TransitionKind::Default: return "default";
  }
  return "?";
}

const ChartState *Chart::findState(const std::string &Id) const {
  auto It = States.find(Id);
  return It == States.end() ? nullptr : &It->second;
}

const ChartJunction *Chart::findJunction(const std::string &Id) const {
  auto It = Junctions.find(Id);
  return It == Junctions.end() ? nullptr : &It->second;
}

const Chart *ChartModel::findChart(const std::string &Name) const {
  for (auto &C : Charts)
    if (C.Name == Name) return &C;
  return nullptr;
}

namespace {

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

std::string trim(std::string_view S) {
  size_t B = 0, E = S.size();
  while (B < E && std::isspace(static_cast<unsigned char>(S[B]))) ++B;
  while (E > B && std::isspace(static_cast<unsigned char>(S[E - 1]))) --E;
  return std::string(S.substr(B, E - B));
}

bool isIdStart(char C) {
  return std::isalpha(static_cast<unsigned char>(C)) || C == '_';
}

bool isIdCont(char C) {
  return std::isalnum(static_cast<unsigned char>(C)) || C == '_';
}

// Read a balanced-bracket region starting at Src[Pos] (which must be
// `Open`). Returns the slice between the brackets (exclusive) and
// advances Pos past the closing `Close`. Sets `Failed = true` when
// the brackets don't balance — label parsing is best-effort, the
// caller decides whether to surface a diagnostic.
std::string readBalanced(const std::string &Src, size_t &Pos, char Open,
                         char Close, bool &Failed) {
  if (Pos >= Src.size() || Src[Pos] != Open) {
    Failed = true;
    return {};
  }
  ++Pos;
  size_t Start = Pos;
  int Depth = 1;
  while (Pos < Src.size()) {
    char C = Src[Pos];
    if (C == Open) ++Depth;
    else if (C == Close) {
      --Depth;
      if (Depth == 0) {
        std::string Out(Src.substr(Start, Pos - Start));
        ++Pos;
        return Out;
      }
    } else if (C == '\'' || C == '"') {
      // Skip MATLAB strings naively — a balanced quote up to the
      // matching delimiter.  Enough for typical Stateflow guards
      // (`x == 'a'`) without trying to be a full MATLAB lexer.
      char Q = C;
      ++Pos;
      while (Pos < Src.size() && Src[Pos] != Q) ++Pos;
      if (Pos < Src.size()) ++Pos;
      continue;
    }
    ++Pos;
  }
  Failed = true;
  return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// TransitionLabel parser.
//
//   label := [event] ['[' guard ']'] ['{' condAction '}'] ['/' transAction]
//
// All four fields are optional. A label that is a single MATLAB
// statement (no leading event/guard/condAction) is treated as
// `transAction` — this is Stateflow's convention for unconditional
// flow-action labels and matches the common transitiveAction-only
// shorthand the IDE writes.
//===----------------------------------------------------------------------===//

TransitionLabel parseTransitionLabel(const std::string &Raw,
                                     SourceLocation Loc) {
  TransitionLabel L;
  L.Raw = Raw;
  L.Loc = Loc;
  std::string S = trim(Raw);
  if (S.empty()) return L;

  size_t Pos = 0;

  // Event = leading identifier (Stateflow allows `E.payload` etc. —
  // we accept identifier + optional `.field`/`(args)` until the
  // first `[`/`{`/`/` or whitespace+anything.
  if (Pos < S.size() && isIdStart(S[Pos])) {
    size_t Start = Pos;
    while (Pos < S.size() && (isIdCont(S[Pos]) || S[Pos] == '.')) ++Pos;
    // Allow a single (...) argument list right after the identifier,
    // e.g. `tick(payload)`.
    if (Pos < S.size() && S[Pos] == '(') {
      int Depth = 1; ++Pos;
      while (Pos < S.size() && Depth > 0) {
        if (S[Pos] == '(') ++Depth;
        else if (S[Pos] == ')') --Depth;
        ++Pos;
      }
    }
    L.Event = trim(S.substr(Start, Pos - Start));
  }

  // Tolerate whitespace between sections.
  while (Pos < S.size() && std::isspace(static_cast<unsigned char>(S[Pos])))
    ++Pos;

  // Guard = `[ ... ]`.
  if (Pos < S.size() && S[Pos] == '[') {
    bool Failed = false;
    std::string Body = readBalanced(S, Pos, '[', ']', Failed);
    if (!Failed) L.Guard = trim(Body);
    else { L.TransAction = trim(S.substr(Pos)); return L; }
    while (Pos < S.size() && std::isspace(static_cast<unsigned char>(S[Pos])))
      ++Pos;
  }

  // CondAction = `{ ... }`.
  if (Pos < S.size() && S[Pos] == '{') {
    bool Failed = false;
    std::string Body = readBalanced(S, Pos, '{', '}', Failed);
    if (!Failed) L.CondAction = trim(Body);
    else { L.TransAction = trim(S.substr(Pos)); return L; }
    while (Pos < S.size() && std::isspace(static_cast<unsigned char>(S[Pos])))
      ++Pos;
  }

  // TransAction = `/ ...` (rest of the label).
  if (Pos < S.size() && S[Pos] == '/') {
    ++Pos;
    L.TransAction = trim(S.substr(Pos));
    return L;
  }

  // No event-prefix and no bracketed section consumed: the entire
  // label is a transition-action shorthand.
  if (L.Event.empty() && L.Guard.empty() && L.CondAction.empty() &&
      Pos < S.size()) {
    L.TransAction = trim(S.substr(Pos));
  }
  return L;
}

namespace {

//===----------------------------------------------------------------------===//
// Builder — FlowDoc → ChartModel.
//===----------------------------------------------------------------------===//

class Builder {
public:
  Builder(DiagnosticEngine &Diag) : Diag_(Diag) {}

  std::optional<ChartModel> build(const flowchart::FlowDoc &Doc) {
    if (!Doc.isStateChart()) {
      // Called from a wrong dispatch — fail loud rather than silently
      // returning an empty model.
      Diag_.error(SourceLocation{},
                  "buildChartModel expects settings.kind=state_chart");
      return std::nullopt;
    }
    ChartModel M;
    M.EntryName = Doc.Entry;
    for (auto &F : Doc.Flows) {
      auto C = buildOne(F);
      if (!C) return std::nullopt;
      M.Charts.push_back(std::move(*C));
    }
    return M;
  }

private:
  DiagnosticEngine &Diag_;

  static JunctionKind junctionKindFromString(const std::string &K) {
    if (K == "junction_history")    return JunctionKind::History;
    if (K == "junction_entry")      return JunctionKind::Entry;
    if (K == "junction_exit")       return JunctionKind::Exit;
    if (K == "junction_default")    return JunctionKind::Default;
    // junction_connective and any future generic flow junction.
    return JunctionKind::Connective;
  }

  static Decomposition parseDecomp(const std::string *S) {
    if (!S) return Decomposition::Leaf;
    if (*S == "or")  return Decomposition::Or;
    if (*S == "and") return Decomposition::And;
    return Decomposition::Leaf;
  }

  static ContainerStyle parseContainer(const std::string *S) {
    if (!S) return ContainerStyle::State;
    if (*S == "subchart") return ContainerStyle::Subchart;
    if (*S == "box")      return ContainerStyle::Box;
    return ContainerStyle::State;
  }

  static bool paramTrue(const flowchart::Node &N, std::string_view Key) {
    auto *S = N.getParam(Key);
    return S && (*S == "true" || *S == "1");
  }

  Action makeAction(const flowchart::Node &N, std::string_view Key) {
    Action A;
    if (auto *S = N.getData(Key)) {
      A.Source = *S;
      auto It = N.DataLocs.find(std::string(Key));
      if (It != N.DataLocs.end()) A.Loc = It->second;
      else                        A.Loc = N.Loc;
    }
    return A;
  }

  std::optional<Chart> buildOne(const flowchart::Flow &F) {
    Chart C;
    C.Name    = F.Name;
    C.Sig     = F.Sig;
    C.Symbols = F.Symbols;
    C.Loc     = F.Loc;

    // First pass: materialise every Node into either a state or a
    // junction. Unknown kinds error out so a typo in the IDE doesn't
    // silently lose a node.
    for (auto &N : F.Nodes) {
      if (N.Kind == "state") {
        ChartState S;
        S.Id    = N.Id;
        S.Label = N.Label;
        S.ParentId = N.Parent;
        S.Decomp    = parseDecomp(N.getParam("decomposition"));
        S.Container = parseContainer(N.getParam("containerStyle"));
        S.IsInitial = paramTrue(N, "isInitial");
        S.HasHistory = paramTrue(N, "hasHistory");
        S.Atomic    = paramTrue(N, "atomic");
        if (auto *O = N.getParam("executionOrder")) {
          try { S.ExecutionOrder = std::stoi(*O); }
          catch (...) { /* Loader already validated this */ }
        }
        S.Entry  = makeAction(N, "entryAction");
        S.During = makeAction(N, "duringAction");
        S.Exit   = makeAction(N, "exitAction");
        // mStateflow on-event handlers — preserved in declaration
        // order. The Loader pulls them out of `data.onEventActions`.
        for (auto &Pair : N.OnEventActions) {
          Action A;
          A.Source = Pair.second;
          auto It = N.OnEventActionLocs.find(Pair.first);
          A.Loc = (It != N.OnEventActionLocs.end()) ? It->second : N.Loc;
          S.OnEvent.emplace_back(Pair.first, std::move(A));
        }
        S.Loc = N.Loc;
        C.States.emplace(S.Id, std::move(S));
      } else if (N.Kind == "junction_connective" ||
                 N.Kind == "junction_history" ||
                 N.Kind == "junction_entry" ||
                 N.Kind == "junction_exit" ||
                 N.Kind == "junction_default") {
        ChartJunction J;
        J.Id   = N.Id;
        J.Kind = junctionKindFromString(N.Kind);
        J.ParentId = N.Parent;
        J.Loc  = N.Loc;
        C.Junctions.emplace(J.Id, std::move(J));
      } else if (N.Kind == "chart_fn_matlab") {
        // Tier 8 — inline MATLAB-bodied chart function. The node's
        // params carry the function body + signature; we emit it
        // as a sibling MATLAB function in the lowering's output
        // so action bodies can call it by name. The call-site
        // itself isn't a state — it's purely metadata.
        ChartFunction F;
        F.Id = N.Id;
        if (auto *S = N.getParam("functionName")) F.Name = *S;
        if (F.Name.empty()) F.Name = N.Id;
        if (auto *Body = N.getData("body")) F.Body = *Body;
        if (auto *Body = N.getParam("body")) F.Body = *Body;
        auto splitCsv = [](const std::string &S) {
          std::vector<std::string> Out;
          std::string Cur;
          for (char Ch : S) {
            if (Ch == ',' || Ch == ' ' || Ch == '\t') {
              if (!Cur.empty()) { Out.push_back(Cur); Cur.clear(); }
            } else Cur += Ch;
          }
          if (!Cur.empty()) Out.push_back(Cur);
          return Out;
        };
        if (auto *S = N.getParam("inputs"))  F.Inputs  = splitCsv(*S);
        if (auto *S = N.getParam("outputs")) F.Outputs = splitCsv(*S);
        F.Loc = N.Loc;
        C.Functions.push_back(std::move(F));
      } else if (N.Kind == "chart_fn_graphical" ||
                 N.Kind == "chart_fn_truth_table") {
        // Graphical functions (control-flow sub-Flow) and truth
        // tables still need the sub-Flow inlining + over/
        // underspecification diagnostic. Surface them as opaque
        // atomic states for now + warn.
        ChartState S;
        S.Id = N.Id;
        S.Label = N.Label;
        S.ParentId = N.Parent;
        S.Decomp = Decomposition::Leaf;
        S.Atomic = true;
        S.Loc = N.Loc;
        Diag_.warning(N.Loc,
                      "chart function call-site \"" + N.Id + "\" (kind \"" +
                          N.Kind + "\") lowered as opaque atomic state — "
                          "graphical / truth-table inlining pending");
        C.States.emplace(S.Id, std::move(S));
      } else if (N.Kind == "comment") {
        // Plain annotation — already legal in control / signal flows.
        continue;
      } else {
        Diag_.error(N.Loc,
                    "node kind \"" + N.Kind +
                        "\" is not valid in a state_chart flow");
        return std::nullopt;
      }
    }

    // Wire parent-child sibling lists for both states and junctions.
    // Source order is preserved by iterating `F.Nodes` in declaration
    // order; that way the dump and downstream codegen come out
    // deterministically.
    for (auto &N : F.Nodes) {
      ChartState    *Parent = nullptr;
      if (!N.Parent.empty()) {
        auto It = C.States.find(N.Parent);
        if (It != C.States.end()) Parent = &It->second;
      }
      auto SIt = C.States.find(N.Id);
      auto JIt = C.Junctions.find(N.Id);
      if (SIt != C.States.end()) {
        if (Parent) Parent->ChildStateIds.push_back(N.Id);
        else        C.RootStateIds.push_back(N.Id);
      } else if (JIt != C.Junctions.end()) {
        if (Parent) Parent->ChildJunctionIds.push_back(N.Id);
        else        C.RootJunctionIds.push_back(N.Id);
      }
    }

    // Edit-time lint: scan every action/guard body and warn when it
    // references an identifier that's not in the chart's symbol
    // table, the signature, a built-in, or a MATLAB keyword. Tier-1
    // UI's "undefined symbol references" red squiggle reads these
    // warnings via -dump-chart; the chart still compiles + runs so
    // partial edits don't block the user.
    {
      std::unordered_set<std::string> Known;
      auto bag = [&](auto &Vec) {
        for (auto &S : Vec) Known.insert(S.Name);
      };
      bag(F.Symbols.Data);
      bag(F.Symbols.Events);
      bag(F.Symbols.Messages);
      for (auto &N : F.Sig.Inputs)  Known.insert(N);
      for (auto &N : F.Sig.Outputs) Known.insert(N);
      // MATLAB keywords + built-in identifiers the rewriter / interp
      // treat specially. Anything else encountered as an identifier
      // in an action triggers a warning.
      static const std::unordered_set<std::string> Builtins{
        "true", "false", "Inf", "NaN", "pi", "eps",
        "abs", "min", "max", "floor", "ceil", "round", "mod",
        "sqrt", "sin", "cos", "exp", "log",
        "after", "before", "every", "at", "in", "emit",
        "temporalCount", "duration",
        "true", "false",
        "if", "else", "elseif", "end", "for", "while", "switch",
        "case", "otherwise", "return", "break", "continue",
        "function", "global", "persistent",
      };
      auto scan = [&](const std::string &Src, SourceLocation Loc,
                      const std::string &Where) {
        size_t I = 0;
        char Quote = 0;
        bool PrevDot = false;
        while (I < Src.size()) {
          char C = Src[I];
          if (Quote) {
            if (C == Quote) Quote = 0;
            ++I; continue;
          }
          if (C == '\'' || C == '"') { Quote = C; ++I; continue; }
          if (C == '%') { while (I < Src.size() && Src[I] != '\n') ++I; continue; }
          if (C == '.' && I + 1 < Src.size() && Src[I+1] != '.') {
            ++I; PrevDot = true; continue;
          }
          if (std::isalpha(static_cast<unsigned char>(C)) || C == '_') {
            size_t Start = I;
            while (I < Src.size() &&
                   (std::isalnum(static_cast<unsigned char>(Src[I])) ||
                    Src[I] == '_')) ++I;
            std::string Id = Src.substr(Start, I - Start);
            if (!PrevDot && !Known.count(Id) && !Builtins.count(Id) &&
                !std::isdigit(static_cast<unsigned char>(Id[0]))) {
              Diag_.warning(Loc, "undefined identifier \"" + Id +
                                     "\" in " + Where);
            }
            PrevDot = false;
            continue;
          }
          if (!std::isspace(static_cast<unsigned char>(C))) PrevDot = false;
          ++I;
        }
      };
      for (auto &N : F.Nodes) {
        if (N.Kind != "state") continue;
        if (auto *S = N.getData("entryAction"))
          scan(*S, N.Loc, "entry action of \"" + N.Id + "\"");
        if (auto *S = N.getData("duringAction"))
          scan(*S, N.Loc, "during action of \"" + N.Id + "\"");
        if (auto *S = N.getData("exitAction"))
          scan(*S, N.Loc, "exit action of \"" + N.Id + "\"");
        for (auto &P : N.OnEventActions)
          scan(P.second, N.Loc,
               "on(" + P.first + ") handler of \"" + N.Id + "\"");
      }
      for (auto &E : F.Edges) {
        if (E.Kind != "transition" || E.Label.empty()) continue;
        TransitionLabel L = parseTransitionLabel(E.Label, E.LabelLoc);
        if (!L.Guard.empty())
          scan(L.Guard, E.Loc, "guard of transition \"" + E.Id + "\"");
        if (!L.CondAction.empty())
          scan(L.CondAction, E.Loc,
               "condition action of transition \"" + E.Id + "\"");
        if (!L.TransAction.empty())
          scan(L.TransAction, E.Loc,
               "transition action of \"" + E.Id + "\"");
      }
    }
    // Edges → Transitions. State-chart edges must carry `kind ==
    // "transition"`; anything else is rejected so the IDE / hand-
    // editor catches typos early.
    for (auto &E : F.Edges) {
      if (E.Kind != "transition") {
        Diag_.error(E.Loc,
                    "edge kind \"" + E.Kind +
                        "\" not valid in a state_chart flow (expected "
                        "\"transition\")");
        return std::nullopt;
      }
      Transition T;
      T.Id       = E.Id;
      T.SourceId = E.From.Node;
      T.DestId   = E.To.Node;
      T.Loc      = E.Loc;
      // Source / dest must resolve to a state or a junction.
      auto exists = [&](const std::string &Id) {
        return C.States.count(Id) || C.Junctions.count(Id);
      };
      if (!exists(T.SourceId)) {
        Diag_.error(E.From.Loc, "transition source \"" + T.SourceId +
                                    "\" is not a chart node");
        return std::nullopt;
      }
      if (!exists(T.DestId)) {
        Diag_.error(E.To.Loc, "transition destination \"" + T.DestId +
                                  "\" is not a chart node");
        return std::nullopt;
      }
      // Optional params.priority / params.kind.
      if (auto It = E.Params.find("priority"); It != E.Params.end()) {
        try { T.Priority = std::stoi(It->second); }
        catch (...) {
          auto LIt = E.ParamLocs.find("priority");
          Diag_.error(LIt != E.ParamLocs.end() ? LIt->second : E.Loc,
                      "transition \"" + T.Id +
                          "\" has non-integer priority \"" + It->second + "\"");
          return std::nullopt;
        }
      }
      if (auto It = E.Params.find("kind"); It != E.Params.end()) {
        if      (It->second == "outer")   T.Kind = TransitionKind::Outer;
        else if (It->second == "inner")   T.Kind = TransitionKind::Inner;
        else if (It->second == "default") T.Kind = TransitionKind::Default;
        else {
          auto LIt = E.ParamLocs.find("kind");
          Diag_.error(LIt != E.ParamLocs.end() ? LIt->second : E.Loc,
                      "transition \"" + T.Id +
                          "\" has unknown kind \"" + It->second +
                          "\" (expected \"outer\", \"inner\", or \"default\")");
          return std::nullopt;
        }
      }
      // Promote a `junction_default`-sourced transition to Default
      // when the IDE forgot the explicit param.
      if (T.Kind == TransitionKind::Outer) {
        auto JIt = C.Junctions.find(T.SourceId);
        if (JIt != C.Junctions.end() &&
            JIt->second.Kind == JunctionKind::Default) {
          T.Kind = TransitionKind::Default;
        }
      }
      T.Label = parseTransitionLabel(E.Label, E.LabelLoc.Offset
                                                  ? E.LabelLoc
                                                  : E.Loc);
      C.Transitions.push_back(std::move(T));
    }

    return C;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public entry point.
//===----------------------------------------------------------------------===//

std::optional<ChartModel> buildChartModel(const flowchart::FlowDoc &Doc,
                                          DiagnosticEngine &Diag) {
  Builder B(Diag);
  return B.build(Doc);
}

//===----------------------------------------------------------------------===//
// Dump.
//===----------------------------------------------------------------------===//

namespace {

void dumpAction(std::ostream &OS, const std::string &Pad, const char *Tag,
                const Action &A) {
  if (A.empty()) return;
  OS << Pad << "." << Tag << "=" << A.Source << "\n";
}

void dumpState(std::ostream &OS, const Chart &C, const std::string &Id,
               int Indent) {
  const ChartState *S = C.findState(Id);
  if (!S) return;
  std::string Pad(Indent * 2, ' ');
  std::string InnerPad((Indent + 1) * 2, ' ');
  OS << Pad << "State " << S->Id;
  if (!S->Label.empty()) OS << " label=" << S->Label;
  OS << " decomp=" << decompositionName(S->Decomp);
  if (S->Container != ContainerStyle::State)
    OS << " container=" << containerStyleName(S->Container);
  if (S->IsInitial)  OS << " initial";
  if (S->HasHistory) OS << " history";
  if (S->Atomic)     OS << " atomic";
  if (S->ExecutionOrder) OS << " exec=" << *S->ExecutionOrder;
  OS << "\n";
  dumpAction(OS, InnerPad, "entry",  S->Entry);
  dumpAction(OS, InnerPad, "during", S->During);
  dumpAction(OS, InnerPad, "exit",   S->Exit);
  for (auto &E : S->OnEvent)
    OS << InnerPad << ".on(" << E.first << ")=" << E.second.Source << "\n";
  for (auto &Cid : S->ChildStateIds)
    dumpState(OS, C, Cid, Indent + 1);
  for (auto &Jid : S->ChildJunctionIds) {
    const ChartJunction *J = C.findJunction(Jid);
    if (!J) continue;
    OS << InnerPad << "Junction " << J->Id << " kind="
       << junctionKindName(J->Kind) << "\n";
  }
}

} // namespace

void dumpChartModel(std::ostream &OS, const ChartModel &M) {
  OS << "ChartModel entry=" << M.EntryName << " charts=" << M.Charts.size()
     << "\n";
  for (auto &C : M.Charts) {
    OS << "Chart " << C.Name;
    if (!C.Sig.Inputs.empty() || !C.Sig.Outputs.empty()) {
      OS << " sig=(";
      for (size_t I = 0; I < C.Sig.Inputs.size(); ++I) {
        if (I) OS << ",";
        OS << C.Sig.Inputs[I];
      }
      OS << ")->(";
      for (size_t I = 0; I < C.Sig.Outputs.size(); ++I) {
        if (I) OS << ",";
        OS << C.Sig.Outputs[I];
      }
      OS << ")";
    }
    OS << " maxIter=" << C.MaxIterations << "\n";
    auto dumpSym = [&](const char *Tag, const flowchart::Symbol &S) {
      OS << "  symbol." << Tag << " name=" << S.Name;
      if (!S.Scope.empty())   OS << " scope=" << S.Scope;
      if (!S.Type.empty())    OS << " type=" << S.Type;
      if (!S.Units.empty())   OS << " units=" << S.Units;
      if (!S.Initial.empty()) OS << " initial=" << S.Initial;
      if (!S.Trigger.empty()) OS << " trigger=" << S.Trigger;
      OS << "\n";
    };
    for (auto &S : C.Symbols.Data)     dumpSym("data",    S);
    for (auto &S : C.Symbols.Events)   dumpSym("event",   S);
    for (auto &S : C.Symbols.Messages) dumpSym("message", S);
    for (auto &Id : C.RootStateIds)    dumpState(OS, C, Id, 1);
    for (auto &Jid : C.RootJunctionIds) {
      const ChartJunction *J = C.findJunction(Jid);
      if (!J) continue;
      OS << "  Junction " << J->Id << " kind=" << junctionKindName(J->Kind)
         << "\n";
    }
    for (auto &T : C.Transitions) {
      OS << "  Transition " << T.Id << " " << T.SourceId << " -> "
         << T.DestId << " kind=" << transitionKindName(T.Kind)
         << " priority=" << T.Priority;
      if (!T.Label.Event.empty())      OS << " event=" << T.Label.Event;
      if (!T.Label.Guard.empty())      OS << " guard=" << T.Label.Guard;
      if (!T.Label.CondAction.empty()) OS << " condA=" << T.Label.CondAction;
      if (!T.Label.TransAction.empty())OS << " transA=" << T.Label.TransAction;
      OS << "\n";
    }
  }
}

} // namespace matlab::statechart
