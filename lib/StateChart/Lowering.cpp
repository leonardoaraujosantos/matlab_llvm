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
// Identifier-aware action / guard rewriter.
//
// The chart-IR keeps every action body as RAW MATLAB source. Before
// emitting MATLAB code we need every reference to a chart symbol
// (`temp`, `heat`, `credit`, ...) rewritten to `state.locals.X` so
// the generated `chart_tick` function actually mutates the state
// struct. We also rewrite event identifiers to `state.events.X`.
//
// The rewriter is a small scanner — not a parser. It tokenises the
// source into either an "identifier run" or a "non-identifier run"
// (string literals + everything else) and only rewrites identifier
// runs that aren't preceded by '.' (i.e. not a struct-field access)
// and aren't MATLAB keywords.
//===----------------------------------------------------------------------===//

const std::set<std::string> &matlabKeywords() {
  static const std::set<std::string> K{
      "break",      "case",     "catch", "classdef", "continue", "else",
      "elseif",     "end",      "for",   "function", "global",   "if",
      "otherwise",  "parfor",   "persistent", "return", "spmd",   "switch",
      "try",        "while",    "true",  "false",    "Inf",      "NaN",
      "pi",         "eps"};
  return K;
}

struct SymbolMap {
  std::unordered_set<std::string> Locals;   // data + signature inputs/outputs
  std::unordered_set<std::string> Events;
  // Chart-scoped helper name (e.g. `myc_in_active_`) — emitted as a
  // sibling function in the lowering's output; the rewriter swaps
  // `in(X)` calls for `<chart>_in_active_(state, 'X')`.
  std::string InHelper;
  // Owning state id for the body currently being rewritten. Empty
  // for chart-root-scoped expressions (e.g. default-junction
  // condition actions). Temporal operators
  // (`after`/`before`/`every`/`at`) look this up to compute
  // (tick_count - entry_times.<owner>).
  std::string OwnerStateId;
};

bool isIdStart(char C) {
  return std::isalpha(static_cast<unsigned char>(C)) || C == '_';
}
bool isIdCont(char C) {
  return std::isalnum(static_cast<unsigned char>(C)) || C == '_';
}

std::string rewriteActionSource(const std::string &Src,
                                const SymbolMap &Syms) {
  std::string Out;
  Out.reserve(Src.size() + 16);
  size_t I = 0;
  char Quote = 0;
  bool PrevDot = false;
  while (I < Src.size()) {
    char C = Src[I];
    if (Quote) {
      Out += C;
      // Two consecutive single quotes inside a single-quoted string
      // escape the quote (MATLAB convention).
      if (C == Quote) {
        if (Quote == '\'' && I + 1 < Src.size() && Src[I + 1] == '\'') {
          Out += Src[I + 1];
          I += 2;
          continue;
        }
        Quote = 0;
      }
      ++I;
      PrevDot = false;
      continue;
    }
    if (C == '\'' || C == '"') {
      // A leading "'" right after an identifier/number/closing-paren
      // is the transpose operator, not a string. Heuristic: a string
      // opener follows whitespace, `=`, `(`, `[`, `{`, `,`, `;`, or
      // the start of the source.
      bool IsString = true;
      if (C == '\'' && !Out.empty()) {
        char Last = Out.back();
        if (std::isalnum(static_cast<unsigned char>(Last)) || Last == '_' ||
            Last == ')' || Last == ']' || Last == '}' || Last == '.')
          IsString = false;
      }
      if (IsString) Quote = C;
      Out += C;
      ++I;
      PrevDot = false;
      continue;
    }
    if (C == '.' && I + 1 < Src.size() && Src[I + 1] != '.') {
      // Mark next token as a field access. Two-dot sequences are
      // MATLAB's range / line-continuation — leave PrevDot off then.
      Out += C;
      ++I;
      PrevDot = true;
      continue;
    }
    if (C == '%') {
      // Comment to end of line — copy as-is, skip rewriting.
      while (I < Src.size() && Src[I] != '\n') Out += Src[I++];
      PrevDot = false;
      continue;
    }
    if (isIdStart(C)) {
      size_t Start = I;
      while (I < Src.size() && isIdCont(Src[I])) ++I;
      std::string Id = Src.substr(Start, I - Start);
      // Temporal operators — `after(N, unit)` / `before(N, unit)` /
      // `every(N, unit)` / `at(N, unit)`. Each lowers to an arithmetic
      // check against (state.tick_count - state.entry_times.<owner>).
      // The `sec` unit is aliased to `tick` for now (1 super-step =
      // 1 sec); fine-grained timing comes when the chart gains a
      // configurable dt.
      auto isTemporal = [](const std::string &S) {
        return S == "after" || S == "before" || S == "every" || S == "at";
      };
      if (!PrevDot && isTemporal(Id) && !Syms.OwnerStateId.empty()) {
        size_t P = I;
        while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
        if (P < Src.size() && Src[P] == '(') {
          ++P;
          while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
          // Parse the integer literal.
          size_t NStart = P;
          if (P < Src.size() && (Src[P] == '+' || Src[P] == '-')) ++P;
          while (P < Src.size() &&
                 std::isdigit(static_cast<unsigned char>(Src[P]))) ++P;
          if (P > NStart) {
            std::string Num = Src.substr(NStart, P - NStart);
            // Optional `, unit`.
            while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
            if (P < Src.size() && Src[P] == ',') {
              ++P;
              while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
              while (P < Src.size() && isIdCont(Src[P])) ++P;
              while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
            }
            if (P < Src.size() && Src[P] == ')') {
              std::string Delta = "(state.tick_count - state.entry_times." +
                                  Syms.OwnerStateId + ")";
              std::string Expr;
              if      (Id == "after")  Expr = "(" + Delta + " >= " + Num + ")";
              else if (Id == "before") Expr = "(" + Delta + " <  " + Num + ")";
              else if (Id == "at")     Expr = "(" + Delta + " == " + Num + ")";
              else if (Id == "every")
                Expr = "((" + Delta + " > 0) && (mod(" + Delta + ", " +
                       Num + ") == 0))";
              Out += Expr;
              I = P + 1;
              PrevDot = false;
              continue;
            }
          }
        }
      }
      // Recognise `in(stateId)` / `in('stateId')` / `in("stateId")`
      // and rewrite to a call against the chart's auto-emitted
      // `<chart>_in_active_` helper. The helper does a switch on the
      // queried state's parent region and answers from the live
      // `state.regions.*` slots.
      if (!PrevDot && Id == "in" && !Syms.InHelper.empty()) {
        size_t P = I;
        while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
        if (P < Src.size() && Src[P] == '(') {
          ++P;
          while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
          std::string Name;
          bool Parsed = false;
          if (P < Src.size() && (Src[P] == '\'' || Src[P] == '"')) {
            char Q = Src[P++];
            size_t NStart = P;
            while (P < Src.size() && Src[P] != Q) ++P;
            if (P < Src.size()) {
              Name = Src.substr(NStart, P - NStart);
              ++P;
              Parsed = true;
            }
          } else if (P < Src.size() && isIdStart(Src[P])) {
            size_t NStart = P;
            while (P < Src.size() && isIdCont(Src[P])) ++P;
            Name = Src.substr(NStart, P - NStart);
            Parsed = true;
          }
          while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
          if (Parsed && P < Src.size() && Src[P] == ')') {
            Out += Syms.InHelper;
            Out += "(state, '";
            Out += Name;
            Out += "')";
            I = P + 1;
            PrevDot = false;
            continue;
          }
        }
      }
      // Recognise `emit('X')`, `emit("X")`, and `emit(X)` so charts
      // can broadcast events from inside an action body. We rewrite
      // the whole call site to `state.events.X = true` (a no-op
      // expression in MATLAB statement context — the user typically
      // writes `emit(X);` so the trailing `;` swallows the value).
      if (!PrevDot && Id == "emit") {
        size_t P = I;
        while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
        if (P < Src.size() && Src[P] == '(') {
          ++P;
          while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
          std::string Name;
          bool Parsed = false;
          if (P < Src.size() && (Src[P] == '\'' || Src[P] == '"')) {
            char Q = Src[P++];
            size_t NStart = P;
            while (P < Src.size() && Src[P] != Q) ++P;
            if (P < Src.size()) {
              Name = Src.substr(NStart, P - NStart);
              ++P;
              Parsed = true;
            }
          } else if (P < Src.size() && isIdStart(Src[P])) {
            size_t NStart = P;
            while (P < Src.size() && isIdCont(Src[P])) ++P;
            Name = Src.substr(NStart, P - NStart);
            Parsed = true;
          }
          while (P < Src.size() && std::isspace(static_cast<unsigned char>(Src[P]))) ++P;
          if (Parsed && P < Src.size() && Src[P] == ')') {
            Out += "state.events.";
            Out += Name;
            Out += " = true";
            I = P + 1;
            PrevDot = false;
            continue;
          }
        }
      }
      if (PrevDot || matlabKeywords().count(Id)) {
        Out += Id;
      } else if (Syms.Locals.count(Id)) {
        Out += "state.locals.";
        Out += Id;
      } else if (Syms.Events.count(Id)) {
        Out += "state.events.";
        Out += Id;
      } else {
        Out += Id;
      }
      PrevDot = false;
      continue;
    }
    Out += C;
    ++I;
    if (!std::isspace(static_cast<unsigned char>(C))) PrevDot = false;
  }
  return Out;
}

//===----------------------------------------------------------------------===//
// Region model.
//
// A `Region` here is an OR-decomposed scope: chart-root or any
// compound state with Decomp == Or. AND parents don't get a region;
// their children are themselves regions (visited per super-step in
// executionOrder). Leaves never get a region.
//===----------------------------------------------------------------------===//

struct Region {
  std::string Id;                       // "chart_root" or a state id
  bool IsRoot = false;
  std::vector<std::string> Substates;   // direct OR children (states only)
  std::string Initial;                  // initial substate id (may be empty)
  // Direct AND parents whose every child must be entered when this
  // region enters one of its substates. Filled in `buildRegions`.
};

// Resolve the initial substate for an OR region. Prefer a substate
// flagged `isInitial`, then fall back to the destination of a
// default-junction outgoing transition in this region.
std::string resolveInitial(const Chart &C, const std::string &RegionId,
                           const std::vector<std::string> &Substates) {
  for (auto &Sid : Substates) {
    const auto *S = C.findState(Sid);
    if (S && S->IsInitial) return Sid;
  }
  // Default junctions: a `junction_default` whose parent is this
  // region's id AND which has an outgoing transition pointing at a
  // substate. The Loader allows ≤1 such junction per OR region.
  for (auto &P : C.Junctions) {
    const ChartJunction &J = P.second;
    if (J.Kind != JunctionKind::Default) continue;
    if (J.ParentId != (RegionId == "chart_root" ? std::string() : RegionId)) continue;
    for (auto &T : C.Transitions) {
      if (T.SourceId == J.Id) return T.DestId;
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Code emitter.
//===----------------------------------------------------------------------===//

class Emitter {
public:
  Emitter(const Chart &C, DiagnosticEngine &Diag) : C_(C), Diag_(Diag) {}

  std::optional<LoweringResult> emit() {
    buildSymbolMap();
    buildRegions();
    if (Failed_) return std::nullopt;
    bucketTransitions();
    LoweringResult R;
    R.InitFunction = sanitize(C_.Name) + "_init";
    R.TickFunction = sanitize(C_.Name) + "_tick";

    std::ostringstream OS;
    emitHeader(OS);
    emitInit(OS);
    emitTick(OS);
    for (auto &Rg : Regions_) emitRegionFn(OS, Rg);
    emitEnterEntryHelpers(OS);
    emitInHelper(OS);
    emitChartFunctions(OS);
    if (Failed_) return std::nullopt;
    R.MatlabSource = OS.str();
    return R;
  }

private:
  const Chart &C_;
  [[maybe_unused]] DiagnosticEngine &Diag_;
  bool Failed_ = false;

  SymbolMap Syms_;
  std::vector<Region> Regions_;
  std::unordered_map<std::string, size_t> RegionByParent_; // parent id → idx

  // Per-region transition lists indexed by source state id. The
  // outer key is the parent id (region id; "" for root). The inner
  // map keys are source state ids; the values are transitions
  // sorted by priority.
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::vector<Transition>>>
      TransByRegion_;
  // Transitions sourced from a junction_default (used during entry).
  std::unordered_map<std::string, std::vector<Transition>> DefaultsByRegion_;

  static std::string sanitize(const std::string &S) {
    std::string Out;
    Out.reserve(S.size());
    for (char C : S) Out += isIdCont(C) ? C : '_';
    if (!Out.empty() && std::isdigit(static_cast<unsigned char>(Out.front())))
      Out.insert(Out.begin(), '_');
    return Out.empty() ? "chart" : Out;
  }

  void buildSymbolMap() {
    for (auto &S : C_.Symbols.Data)     Syms_.Locals.insert(S.Name);
    for (auto &N : C_.Sig.Inputs)       Syms_.Locals.insert(N);
    for (auto &N : C_.Sig.Outputs)      Syms_.Locals.insert(N);
    for (auto &S : C_.Symbols.Events)   Syms_.Events.insert(S.Name);
    Syms_.InHelper = sanitize(C_.Name) + "_in_active_";
  }

  void buildRegions() {
    // Chart-root region: the OR scope holding chart-level states.
    Region Root;
    Root.Id = "chart_root";
    Root.IsRoot = true;
    for (auto &Sid : C_.RootStateIds) {
      const auto *S = C_.findState(Sid);
      if (!S) continue;
      Root.Substates.push_back(Sid);
    }
    Root.Initial = resolveInitial(C_, "chart_root", Root.Substates);
    if (Root.Initial.empty() && !Root.Substates.empty()) {
      // No isInitial / no default junction — pick the first.
      Root.Initial = Root.Substates.front();
    }
    Regions_.push_back(Root);
    RegionByParent_[""] = 0;

    // Walk every compound state. OR parents become regions; AND
    // parents are visited as containers (their children may
    // themselves be regions, handled recursively).
    std::vector<std::string> Stack;
    for (auto &Sid : C_.RootStateIds) Stack.push_back(Sid);
    while (!Stack.empty()) {
      std::string Id = Stack.back(); Stack.pop_back();
      const auto *S = C_.findState(Id);
      if (!S) continue;
      if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
        Region R;
        R.Id = Id;
        R.Substates = S->ChildStateIds;
        R.Initial = resolveInitial(C_, Id, R.Substates);
        if (R.Initial.empty() && !R.Substates.empty())
          R.Initial = R.Substates.front();
        RegionByParent_[Id] = Regions_.size();
        Regions_.push_back(R);
      } else if (S->Decomp == Decomposition::And) {
        // Children must declare execution order (Loader guarantees
        // this). Sort them now so the emit pass walks them in
        // priority order.
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
        AndChildren_[Id] = Kids;
      }
      for (auto &Cid : S->ChildStateIds) Stack.push_back(Cid);
    }
  }

  std::unordered_map<std::string, std::vector<std::string>> AndChildren_;

  void bucketTransitions() {
    for (auto &T : C_.Transitions) {
      // Determine owning region: parent of the source state (or
      // parent of the junction the transition sources from for
      // default-junction-rooted transitions).
      std::string Parent;
      if (auto *S = C_.findState(T.SourceId))    Parent = S->ParentId;
      else if (auto *J = C_.findJunction(T.SourceId)) Parent = J->ParentId;
      else continue;
      // Source = junction_default belongs to the entry path; bucket
      // separately.
      const auto *J = C_.findJunction(T.SourceId);
      if (J && J->Kind == JunctionKind::Default) {
        DefaultsByRegion_[Parent].push_back(T);
      } else {
        TransByRegion_[Parent][T.SourceId].push_back(T);
      }
    }
    // Sort each per-source transition list by priority (stable).
    for (auto &P : TransByRegion_)
      for (auto &Q : P.second)
        std::stable_sort(Q.second.begin(), Q.second.end(),
                         [](const Transition &A, const Transition &B) {
                           return A.Priority < B.Priority;
                         });
  }

  std::string rewrite(const std::string &Src) {
    return rewriteActionSource(Src, Syms_);
  }
  // Temporary owner-state setter for the duration of an emit. Used
  // by the temporal-operator rewriter to compute (tick_count -
  // entry_times.<owner>).
  struct OwnerScope {
    SymbolMap &Syms;
    std::string Prev;
    OwnerScope(SymbolMap &S, std::string New) : Syms(S), Prev(S.OwnerStateId) {
      S.OwnerStateId = std::move(New);
    }
    ~OwnerScope() { Syms.OwnerStateId = std::move(Prev); }
  };

  //===-------- emit helpers ------------------------------------------------===//

  void emitHeader(std::ostream &OS) {
    OS << "% Auto-generated by matlabc — DO NOT EDIT.\n";
    OS << "% Chart: " << C_.Name << "\n\n";
  }

  void emitInit(std::ostream &OS) {
    OS << "function state = " << sanitize(C_.Name) << "_init()\n";
    OS << "  state = struct();\n";
    OS << "  state.locals = struct();\n";
    for (auto &S : C_.Symbols.Data) {
      OS << "  state.locals." << S.Name << " = ";
      OS << (S.Initial.empty() ? "0" : rewrite(S.Initial)) << ";\n";
    }
    // Inputs that aren't in the data table still need slots in state.locals.
    std::unordered_set<std::string> SeenData;
    for (auto &S : C_.Symbols.Data) SeenData.insert(S.Name);
    for (auto &N : C_.Sig.Inputs) {
      if (SeenData.count(N)) continue;
      OS << "  state.locals." << N << " = 0;\n";
    }
    for (auto &N : C_.Sig.Outputs) {
      if (SeenData.count(N)) continue;
      OS << "  state.locals." << N << " = 0;\n";
    }
    OS << "  state.regions = struct();\n";
    for (auto &Rg : Regions_) {
      OS << "  state.regions." << sanitize(Rg.Id) << " = '';\n";
    }
    // History slots: one per OR-decomposed parent whose `hasHistory`
    // is set. Stored alongside the region vector; the entry chain
    // (see emitEnterState) reads them when re-entering a history
    // parent, the transition firing path writes them when exiting a
    // history parent. Always initialised to empty so first-entry
    // behaves like the regular "initial substate" path.
    OS << "  state.history = struct();\n";
    for (auto &P : C_.States) {
      const ChartState &S = P.second;
      if (!S.HasHistory) continue;
      OS << "  state.history." << sanitize(S.Id) << " = '';\n";
    }
    OS << "  state.events = struct();\n";
    for (auto &S : C_.Symbols.Events) {
      OS << "  state.events." << S.Name << " = false;\n";
    }
    // Temporal-operator state: tick counter advances once per
    // super-step; entry_times records the tick value at the moment
    // each state was entered (used by `after`/`before`/`every`/`at`).
    OS << "  state.tick_count = 0;\n";
    OS << "  state.entry_times = struct();\n";
    for (auto &P : C_.States) {
      OS << "  state.entry_times." << sanitize(P.first) << " = 0;\n";
    }
    // Auto-snapshot ring — off by default; the IDE flips it on for
    // step-back. Storage initialised so mstateflow_auto_snap can
    // append cheaply.
    OS << "  state.auto_snapshot = false;\n";
    OS << "  state.auto_snaps = {};\n";
    OS << "  state.initialized = false;\n";
    OS << "end\n\n";
  }

  void emitTick(std::ostream &OS) {
    OS << "function [outputs, state] = " << sanitize(C_.Name)
       << "_tick(state, inputs, events)\n";
    OS << "  if ~isstruct(state) || ~isfield(state, 'initialized')\n";
    OS << "    state = " << sanitize(C_.Name) << "_init();\n";
    OS << "  end\n";
    // Copy inputs into state.locals.
    for (auto &N : C_.Sig.Inputs)
      OS << "  state.locals." << N << " = inputs." << N << ";\n";
    // Merge events.
    for (auto &S : C_.Symbols.Events)
      OS << "  if isfield(events, '" << S.Name << "'), state.events."
         << S.Name << " = logical(events." << S.Name << "); end\n";
    // First-tick entry chain: walk default transitions from chart
    // root down.
    OS << "  if ~state.initialized\n";
    OS << "    state = " << sanitize(C_.Name)
       << "_enter_" << sanitize(Regions_[RegionByParent_[""]].Id)
       << "(state);\n";
    OS << "    state.initialized = true;\n";
    OS << "  end\n";
    // Advance the chart-wide tick counter once per super-step
    // invocation so temporal operators (`after`/`before`/`every`/`at`)
    // can compute time-since-entry as a delta.
    OS << "  state.tick_count = state.tick_count + 1;\n";
    // Super-step fixed-point loop. The trailing `~fired` check after
    // the loop body raises a runtime warning when the cap saturated
    // without quiescing — typical signature of a chart with mutually
    // re-triggering transitions (parity with Stateflow §2-41).
    OS << "  fired = true;\n";
    OS << "  for iter = 1:" << C_.MaxIterations << "\n";
    OS << "    if ~fired, break; end\n";
    OS << "    fired = false;\n";
    emitTickStepRoot(OS, "    ");
    OS << "  end\n";
    OS << "  if fired\n";
    OS << "    warning('mstateflow:maxIterations', ...\n";
    OS << "            'chart \"" << C_.Name
       << "\" super-step did not converge within %d iterations', "
       << C_.MaxIterations << ");\n";
    OS << "  end\n";
    // Clear events after super-step (one-shot semantics).
    for (auto &S : C_.Symbols.Events)
      OS << "  state.events." << S.Name << " = false;\n";
    // Auto-snapshot at super-step boundary (gated by
    // `state.auto_snapshot` — off by default so the path is free
    // unless the IDE turns it on for step-back). Tier 6.
    OS << "  state = mstateflow_auto_snap(state);\n";
    // Materialise outputs.
    OS << "  outputs = struct();\n";
    for (auto &N : C_.Sig.Outputs)
      OS << "  outputs." << N << " = state.locals." << N << ";\n";
    // Tier-10 active-state output port: surface the live region
    // vector so an enclosing mflowLink signal-flow document can wire
    // a chart's active configuration into downstream blocks (§11-38).
    // Always emitted — it's a struct copy, zero cost when unused.
    OS << "  outputs.active_state_ = state.regions;\n";
    OS << "end\n\n";
  }

  // Recursive helper: emit the step call for the substate active in
  // `RegionId`. Stepping a leaf does nothing in itself — the leaf's
  // outgoing transitions live in the parent region's dispatch table.
  // Stepping an AND parent walks every child region in order.
  void emitTickStepRoot(std::ostream &OS, const std::string &Pad) {
    OS << Pad << "[state, fired] = " << sanitize(C_.Name)
       << "_region_" << sanitize(Regions_[RegionByParent_[""]].Id)
       << "(state, fired);\n";
  }

  // Emit the region-step function. Recursive entry into the active
  // substate's region happens inline within the switch case so AND
  // parents fan out per super-step iteration.
  void emitRegionFn(std::ostream &OS, const Region &Rg) {
    std::string RegName = sanitize(Rg.Id);
    OS << "function [state, fired] = " << sanitize(C_.Name) << "_region_"
       << RegName << "(state, fired)\n";
    OS << "  active = state.regions." << RegName << ";\n";
    OS << "  switch active\n";
    for (auto &Sid : Rg.Substates) {
      OS << "    case '" << Sid << "'\n";
      emitSubstateBody(OS, Rg, Sid, "      ");
    }
    OS << "    otherwise\n";
    OS << "      % no active substate\n";
    OS << "  end\n";
    OS << "end\n\n";
  }

  // For a given (region, active substate):
  //   1. Try every outgoing transition of this state in priority
  //      order. Each transition is wrapped in `if ~fired && <cond>`;
  //      on fire, sets fired=true. Remaining transitions short-
  //      circuit on `~fired`.
  //   2. The during-action and AND/OR recursion are gated on
  //      `~fired` so a fired transition doesn't also run during.
  //
  // No `return` is used inside the switch — matlabc's MATLAB → MLIR
  // lowering rejects `return` ops nested inside SCF regions; this
  // fired-guard pattern compiles down through `-emit-c` and friends.
  void emitSubstateBody(std::ostream &OS, const Region &Rg,
                        const std::string &Sid, const std::string &Pad) {
    const auto *S = C_.findState(Sid);
    if (!S) return;
    OwnerScope Owner(Syms_, sanitize(Sid));
    std::string ParentKey = (Rg.Id == "chart_root" ? std::string() : Rg.Id);
    auto Bucket = TransByRegion_[ParentKey].find(Sid);
    if (Bucket != TransByRegion_[ParentKey].end()) {
      for (auto &T : Bucket->second) emitTransition(OS, T, Pad);
    }
    // On-event handlers — fire when the named event is set and the
    // super-step iteration hasn't already fired a transition. Run
    // *before* the during action (Stateflow convention §2-37).
    for (auto &OE : S->OnEvent) {
      if (!Syms_.Events.count(OE.first)) continue;
      OS << Pad << "if ~fired && state.events." << OE.first << "\n";
      OS << Pad << "  " << rewrite(OE.second.Source) << ";\n";
      OS << Pad << "end\n";
    }
    // During action — gated so a fired-this-iter transition doesn't
    // also execute the during.
    if (!S->During.empty()) {
      OS << Pad << "if ~fired\n";
      OS << Pad << "  " << rewrite(S->During.Source) << ";\n";
      OS << Pad << "end\n";
    }
    auto It = AndChildren_.find(Sid);
    if (It != AndChildren_.end()) {
      for (auto &Cid : It->second) emitAndChildStep(OS, Cid, Pad);
    } else if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
      OS << Pad << "[state, fired] = " << sanitize(C_.Name) << "_region_"
         << sanitize(Sid) << "(state, fired);\n";
    }
  }

  void emitAndChildStep(std::ostream &OS, const std::string &Cid,
                        const std::string &Pad) {
    const auto *Child = C_.findState(Cid);
    if (!Child) return;
    if (Child->Decomp == Decomposition::Or && !Child->ChildStateIds.empty()) {
      OS << Pad << "[state, fired] = " << sanitize(C_.Name) << "_region_"
         << sanitize(Cid) << "(state, fired);\n";
    } else if (Child->Decomp == Decomposition::And) {
      // Nested AND under AND — recurse on its children.
      auto It = AndChildren_.find(Cid);
      if (It != AndChildren_.end())
        for (auto &K : It->second) emitAndChildStep(OS, K, Pad);
    } else {
      // Leaf child of AND: nothing to step beyond running its
      // during-action.
      if (!Child->During.empty())
        OS << Pad << rewrite(Child->During.Source) << ";\n";
    }
  }

  void emitTransition(std::ostream &OS, const Transition &T,
                      const std::string &Pad) {
    const auto *SrcS  = C_.findState(T.SourceId);
    const auto *DstS  = C_.findState(T.DestId);
    const auto *DstJ  = C_.findJunction(T.DestId);
    // Build guard expression: events.<E> && (<guard>).
    std::string Cond = "true";
    bool HasEvent = !T.Label.Event.empty() &&
                    Syms_.Events.count(T.Label.Event);
    bool HasGuard = !T.Label.Guard.empty();
    if (HasEvent && HasGuard)
      Cond = "state.events." + T.Label.Event + " && (" + rewrite(T.Label.Guard) + ")";
    else if (HasEvent)
      Cond = "state.events." + T.Label.Event;
    else if (HasGuard)
      Cond = "(" + rewrite(T.Label.Guard) + ")";
    // `~fired` guard so once a higher-priority transition has fired
    // earlier in the per-substate dispatch table, the rest sit out
    // this super-step iteration.
    OS << Pad << "if ~fired && " << Cond << "\n";
    // Cond action runs regardless of inner/outer.
    if (!T.Label.CondAction.empty())
      OS << Pad << "  " << rewrite(T.Label.CondAction) << ";\n";
    // Inner transitions don't exit / re-enter the source — they only
    // run the cond + trans actions. Stateflow §1-45.
    bool IsInner = (T.Kind == TransitionKind::Inner);
    if (!IsInner && SrcS && !SrcS->Exit.empty())
      OS << Pad << "  " << rewrite(SrcS->Exit.Source) << ";\n";
    if (!T.Label.TransAction.empty())
      OS << Pad << "  " << rewrite(T.Label.TransAction) << ";\n";
    if (IsInner) {
      OS << Pad << "  fired = true;\n";
      OS << Pad << "end\n";
      return;
    }
    // For super-transitions (src + dst don't share immediate parent)
    // walk every OR ancestor of src above its immediate parent,
    // clearing its region slot + running its exit action, up to the
    // LCA. Then walk LCA→dst ancestors entering each. Sibling
    // transitions skip both walks (LCA == src.parent == dst.parent).
    std::string SrcParent = SrcS ? SrcS->ParentId : std::string();
    std::string DstParent;
    if (DstS) DstParent = DstS->ParentId;
    else if (DstJ) DstParent = DstJ->ParentId;
    if (SrcParent != DstParent) {
      std::string LCA = lcaOf(T.SourceId, DstS ? T.DestId : DstParent);
      // Walk src.parent up to LCA exclusive.
      std::string Cur = SrcParent;
      while (!Cur.empty() && Cur != LCA) {
        const ChartState *Anc = C_.findState(Cur);
        if (Anc) {
          // Save history before clearing the slot.
          if (Anc->HasHistory)
            OS << Pad << "  state.history." << sanitize(Cur)
               << " = state.regions." << sanitize(Cur) << ";\n";
          if (!Anc->Exit.empty())
            OS << Pad << "  " << rewrite(Anc->Exit.Source) << ";\n";
          OS << Pad << "  state.regions." << sanitize(Cur) << " = '';\n";
          Cur = Anc->ParentId;
        } else break;
      }
      // Walk LCA → dst ancestors (above dst's immediate parent),
      // setting each OR parent's region slot + running entry action.
      std::vector<std::string> EnterChain;
      Cur = DstParent;
      while (!Cur.empty() && Cur != LCA) {
        EnterChain.push_back(Cur);
        const ChartState *Anc = C_.findState(Cur);
        if (!Anc) break;
        Cur = Anc->ParentId;
      }
      std::reverse(EnterChain.begin(), EnterChain.end());
      for (auto &Id : EnterChain) {
        const ChartState *Anc = C_.findState(Id);
        if (!Anc) continue;
        const ChartState *AncParent =
            Anc->ParentId.empty() ? nullptr : C_.findState(Anc->ParentId);
        bool ParentHasSlot =
            Anc->ParentId.empty() ||
            (AncParent && AncParent->Decomp == Decomposition::Or);
        if (ParentHasSlot)
          OS << Pad << "  state.regions."
             << sanitize(Anc->ParentId.empty() ? "chart_root" : Anc->ParentId)
             << " = '" << Id << "';\n";
        if (!Anc->Entry.empty())
          OS << Pad << "  " << rewrite(Anc->Entry.Source) << ";\n";
      }
    }
    if (DstS) {
      emitEnterState(OS, T.DestId, Pad + "  ");
    } else if (DstJ) {
      // Connective / entry / exit / default / history junction —
      // recursively emit the chain. The first outgoing whose guard
      // passes commits the path; control falls through to its
      // destination's enter chain. History redirects to the parent
      // state's `state.history.<parent>` slot when set.
      emitJunctionChain(OS, T.DestId, Pad + "  ", /*Depth=*/0);
    }
    // (No explicit `state.regions.<parent> = ...` here — `emitEnterState`
    // writes the destination's parent region slot as it descends, so
    // double-writing would clutter the output without changing
    // semantics.)
    (void)SrcS;
    OS << Pad << "  fired = true;\n";
    OS << Pad << "end\n";
  }

  // Emit a junction chain rooted at JctId. Connective / entry / exit
  // junctions evaluate their outgoing transitions in priority order;
  // the first whose guard passes commits its cond + trans actions
  // and recurses into its destination. History junctions read the
  // parent's `state.history.<parent>` slot. Depth bound keeps a
  // pathological mutually-recursive junction graph from blowing the
  // emitter (defensive — the loader doesn't yet enforce a DAG).
  void emitJunctionChain(std::ostream &OS, const std::string &JctId,
                         const std::string &Pad, int Depth) {
    if (Depth > 16) {
      OS << Pad << "% junction chain depth limit exceeded\n";
      return;
    }
    const auto *J = C_.findJunction(JctId);
    if (!J) return;
    if (J->Kind == JunctionKind::History) {
      OS << Pad << "if isfield(state.history, '" << sanitize(J->ParentId)
         << "') && ~isempty(state.history." << sanitize(J->ParentId)
         << ")\n";
      const auto *Parent = C_.findState(J->ParentId);
      if (Parent) {
        OS << Pad << "  switch state.history." << sanitize(J->ParentId)
           << "\n";
        for (auto &Cid : Parent->ChildStateIds) {
          OS << Pad << "    case '" << Cid << "'\n";
          emitEnterState(OS, Cid, Pad + "      ");
        }
        OS << Pad << "  end\n";
        OS << Pad << "else\n";
        std::string Init = Parent ? findInitialSub(*Parent) : std::string();
        if (!Init.empty()) emitEnterState(OS, Init, Pad + "  ");
        OS << Pad << "end\n";
      }
      return;
    }
    // Collect outgoing transitions, priority-sorted.
    std::vector<const Transition *> Out;
    for (auto &T : C_.Transitions)
      if (T.SourceId == JctId) Out.push_back(&T);
    std::stable_sort(Out.begin(), Out.end(),
        [](const Transition *A, const Transition *B) {
          return A->Priority < B->Priority;
        });
    bool First = true;
    for (auto *T : Out) {
      // Build guard expression. Junction-rooted transitions never
      // carry an event qualifier (the event was already consumed by
      // the head transition that landed us on this junction), so we
      // only look at the guard portion of the label.
      std::string Cond = "true";
      if (!T->Label.Guard.empty())
        Cond = "(" + rewrite(T->Label.Guard) + ")";
      OS << Pad << (First ? "if " : "elseif ") << Cond << "\n";
      First = false;
      if (!T->Label.CondAction.empty())
        OS << Pad << "  " << rewrite(T->Label.CondAction) << ";\n";
      if (!T->Label.TransAction.empty())
        OS << Pad << "  " << rewrite(T->Label.TransAction) << ";\n";
      if (C_.findState(T->DestId)) {
        emitEnterState(OS, T->DestId, Pad + "  ");
      } else {
        emitJunctionChain(OS, T->DestId, Pad + "  ", Depth + 1);
      }
    }
    if (!First) OS << Pad << "end\n";
  }

  // Lowest common ancestor of two state ids. Walks the parent chain
  // of A into a set, then walks B's chain until it hits one. Returns
  // "" for "no common ancestor" (i.e. chart-root).
  std::string lcaOf(const std::string &A, const std::string &B) const {
    auto chainOf = [&](const std::string &Id) {
      std::vector<std::string> Out;
      std::string Cur = Id;
      while (!Cur.empty()) {
        Out.push_back(Cur);
        const ChartState *S = C_.findState(Cur);
        const ChartJunction *J = C_.findJunction(Cur);
        Cur = S ? S->ParentId : (J ? J->ParentId : std::string());
      }
      return Out;
    };
    std::unordered_set<std::string> AS;
    for (auto &S : chainOf(A)) AS.insert(S);
    for (auto &S : chainOf(B)) if (AS.count(S)) return S;
    return {};
  }

  std::string findInitialSub(const ChartState &S) const {
    for (auto &Cid : S.ChildStateIds) {
      const auto *Cs = C_.findState(Cid);
      if (Cs && Cs->IsInitial) return Cid;
    }
    if (!S.ChildStateIds.empty()) return S.ChildStateIds.front();
    return {};
  }

  // Emit the entry chain for a state (recursively into OR / AND
  // substates). For OR parents, descend into the initial substate;
  // for AND parents, enter every child concurrently.
  //
  // Region-slot writes are gated by the *parent's* decomposition:
  //   - chart-root parent → write `state.regions.chart_root = '<Sid>'`.
  //   - OR parent state   → write `state.regions.<parent> = '<Sid>'`.
  //   - AND parent state  → no slot (children are always co-active).
  void emitEnterState(std::ostream &OS, const std::string &Sid,
                      const std::string &Pad) {
    const auto *S = C_.findState(Sid);
    if (!S) return;
    OwnerScope Owner(Syms_, sanitize(Sid));
    const auto *Parent =
        S->ParentId.empty() ? nullptr : C_.findState(S->ParentId);
    bool ParentHasSlot =
        S->ParentId.empty() ||
        (Parent && Parent->Decomp == Decomposition::Or);
    if (ParentHasSlot) {
      std::string ParentRegion =
          S->ParentId.empty() ? "chart_root" : S->ParentId;
      OS << Pad << "state.regions." << sanitize(ParentRegion)
         << " = '" << Sid << "';\n";
    }
    // Stamp the per-state entry-time so temporal operators referenced
    // in this state's body can compute (tick_count - entry_time).
    OS << Pad << "state.entry_times." << sanitize(Sid)
       << " = state.tick_count;\n";
    if (!S->Entry.empty())
      OS << Pad << rewrite(S->Entry.Source) << ";\n";
    if (S->Decomp == Decomposition::Or && !S->ChildStateIds.empty()) {
      auto It = RegionByParent_.find(Sid);
      if (It == RegionByParent_.end()) return;
      const Region &Rg = Regions_[It->second];
      if (S->HasHistory) {
        // Re-enter the substate that was active when this parent was
        // last exited; fall back to the initial substate on first
        // entry (history slot empty).
        OS << Pad << "if isfield(state.history, '" << sanitize(Sid)
           << "') && ~isempty(state.history." << sanitize(Sid) << ")\n";
        OS << Pad << "  switch state.history." << sanitize(Sid) << "\n";
        for (auto &Cid : S->ChildStateIds) {
          OS << Pad << "    case '" << Cid << "'\n";
          emitEnterState(OS, Cid, Pad + "      ");
        }
        OS << Pad << "  end\n";
        OS << Pad << "else\n";
        if (!Rg.Initial.empty()) emitEnterState(OS, Rg.Initial, Pad + "  ");
        OS << Pad << "end\n";
      } else if (!Rg.Initial.empty()) {
        emitEnterState(OS, Rg.Initial, Pad);
      }
    } else if (S->Decomp == Decomposition::And) {
      auto It = AndChildren_.find(Sid);
      if (It != AndChildren_.end())
        for (auto &Cid : It->second) emitEnterState(OS, Cid, Pad);
    }
  }

  // Emit each chart_fn_matlab node's body as a sibling MATLAB
  // function so action bodies that call it by name resolve through
  // the standard matlabc symbol path. Recursion detection deferred
  // to a follow-on slice — matlabc itself rejects infinite recursion
  // at execution time.
  void emitChartFunctions(std::ostream &OS) {
    for (auto &F : C_.Functions) {
      OS << "function ";
      if (!F.Outputs.empty()) {
        OS << "[";
        for (size_t I = 0; I < F.Outputs.size(); ++I) {
          if (I) OS << ", ";
          OS << F.Outputs[I];
        }
        OS << "] = ";
      }
      OS << F.Name << "(";
      for (size_t I = 0; I < F.Inputs.size(); ++I) {
        if (I) OS << ", ";
        OS << F.Inputs[I];
      }
      OS << ")\n";
      if (!F.Body.empty()) OS << "  " << F.Body << "\n";
      OS << "end\n\n";
    }
  }

  // Emit the chart-scoped `<chart>_in_active_(state, id)` helper —
  // resolves an `in(X)` predicate in actions / guards. The IDE-side
  // `setSymbolBreakpoints` path doesn't need this; only the lowered
  // MATLAB path does (the interpreter handles `in()` natively via
  // its builtin dispatch).
  void emitInHelper(std::ostream &OS) {
    OS << "function out = " << Syms_.InHelper << "(state, id)\n";
    OS << "  out = false;\n";
    OS << "  switch id\n";
    for (auto &P : C_.States) {
      const ChartState &S = P.second;
      OS << "    case '" << S.Id << "'\n";
      // Walk up — a state is active iff every ancestor's region slot
      // selects the path to it, except AND parents (always-active
      // children). Emit a chained && of strcmp() checks. For nested
      // states this yields several conjuncts.
      std::vector<std::pair<std::string, std::string>> Checks;
      std::string Cur = S.Id;
      while (true) {
        const ChartState *CurS = C_.findState(Cur);
        if (!CurS) break;
        const std::string Parent = CurS->ParentId;
        const ChartState *ParentS =
            Parent.empty() ? nullptr : C_.findState(Parent);
        bool AndParent =
            ParentS && ParentS->Decomp == Decomposition::And;
        if (!AndParent) {
          std::string Key = Parent.empty() ? "chart_root" : Parent;
          Checks.emplace_back(sanitize(Key), Cur);
        }
        if (Parent.empty()) break;
        Cur = Parent;
      }
      if (Checks.empty()) {
        OS << "      out = true;\n";
      } else {
        OS << "      out = ";
        for (size_t I = 0; I < Checks.size(); ++I) {
          if (I) OS << " && ";
          OS << "strcmp(state.regions." << Checks[I].first
             << ", '" << Checks[I].second << "')";
        }
        OS << ";\n";
      }
    }
    OS << "  end\n";
    OS << "end\n\n";
  }

  // Emit per-region `<chart>_enter_<region>(state)` helper that
  // initialises the region's active slot and runs the initial sub-
  // state's entry chain. Only chart-root needs one for now (the
  // tick function calls it on the first invocation); other regions
  // are entered inline via `emitEnterState`.
  void emitEnterEntryHelpers(std::ostream &OS) {
    const Region &Root = Regions_[RegionByParent_[""]];
    OS << "function state = " << sanitize(C_.Name) << "_enter_"
       << sanitize(Root.Id) << "(state)\n";
    if (!Root.Initial.empty()) {
      // The recursive `emitEnterState` already writes the parent
      // region slot — no need to set it here.
      emitEnterState(OS, Root.Initial, "  ");
    } else {
      OS << "  % chart-root has no initial substate\n";
    }
    OS << "end\n";
  }
};

} // namespace

std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag) {
  Emitter E(C, Diag);
  return E.emit();
}

} // namespace matlab::statechart
