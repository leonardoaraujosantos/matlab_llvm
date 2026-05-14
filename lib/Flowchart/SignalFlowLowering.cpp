#include "matlab/Flowchart/MflowLinkModel.h"
#include "matlab/Flowchart/MflowLinkSim.h"

#include "matlab/Basic/Diagnostic.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

//===----------------------------------------------------------------------===//
// SignalFlowLowering — the signal-flow sibling to GraphToAST.
//
// Control-flow `.mflow` goes Loader → GraphToAST → structured AST →
// backends. A signal-flow `.mflow` takes this path instead: Loader →
// `lowerSignalFlow` → `MflowLinkModel` IR → simulation runtime. There
// is no statement AST — a block diagram has no control flow to
// structure. See `docs/mflow_link_roadmap.md` §6.
//===----------------------------------------------------------------------===//

namespace matlab::flowchart {

namespace {

//===----------------------------------------------------------------------===//
// Block-kind catalogue.
//
// One row per `signal_*` kind. `Supported` is the Tier-B lowering set
// (the IDE-shipped kinds of roadmap §5.2). A `Known` but unsupported
// kind is a *reserved* kind — accepted and round-tripped by the loader,
// rejected here with a sourced diagnostic until its evaluator ships.
//===----------------------------------------------------------------------===//

struct KindInfo {
  bool Known = false;
  bool Supported = false;
  bool Composite = false;        // signal_subsystem / signal_inport / signal_outport
  bool LoopBreakerAlways = false; // integrator / unit_delay / zoh
  bool ZeroCrossing = false;      // switch / saturation
  SampleTimeClass Sample = SampleTimeClass::FixedInMinor;
};

const KindInfo *lookupKind(const std::string &K) {
  static const std::map<std::string, KindInfo> Table = [] {
    std::map<std::string, KindInfo> T;
    auto add = [&](const char *Name, KindInfo I) {
      I.Known = true;
      T[Name] = I;
    };
    const SampleTimeClass FIM = SampleTimeClass::FixedInMinor;
    const SampleTimeClass CONT = SampleTimeClass::Continuous;
    const SampleTimeClass DISC = SampleTimeClass::Discrete;
    const SampleTimeClass CONST = SampleTimeClass::Constant;

    // --- Supported (Tier-B) -------------------------------------------------
    // Sources.
    add("signal_constant", {true, true, false, false, false, CONST});
    add("signal_step",     {true, true, false, false, false, FIM});
    add("signal_sine",     {true, true, false, false, false, FIM});
    add("signal_pulse",    {true, true, false, false, false, FIM});
    add("signal_ramp",     {true, true, false, false, false, FIM});
    add("signal_clock",    {true, true, false, false, false, FIM});
    add("signal_chirp",    {true, true, false, false, false, FIM});
    add("signal_noise",    {true, true, false, false, false, FIM});
    // Tier F carve-out — clock-like pulse generator for driving
    // `signal_triggered_subsystem`. Emits a one-step `1` at every
    // `period` boundary, `0` between. Sample class is fixed-in-
    // minor because the pulse width is exactly one major step;
    // a `Discrete` class would need a sub-step finer than h.
    add("signal_function_call_generator",
                           {true, true, false, false, false, FIM});
    // Sinks.
    add("signal_scope",        {true, true, false, false, false, FIM});
    add("signal_display",      {true, true, false, false, false, FIM});
    add("signal_to_workspace", {true, true, false, false, false, FIM});
    add("signal_terminator",   {true, true, false, false, false, FIM});
    // Continuous.
    add("signal_integrator",   {true, true, false, true,  false, CONT});
    add("signal_derivative",   {true, true, false, false, false, CONT});
    add("signal_transfer_fcn", {true, true, false, false, false, CONT});
    add("signal_state_space",  {true, true, false, false, false, CONT});
    add("signal_zero_pole",    {true, true, false, false, false, CONT});
    add("signal_transport_delay",
                               {true, true, false, true,  false, CONT});
    // Discrete.
    add("signal_unit_delay", {true, true, false, true, false, DISC});
    add("signal_zoh",        {true, true, false, true, false, DISC});
    add("signal_discrete_integrator",
                             {true, true, false, true, false, DISC});
    add("signal_discrete_filter",
                             {true, true, false, true, false, DISC});
    add("signal_rate_transition",
                             {true, true, false, true, false, DISC});
    // Lookup tables (Tier H — table-driven scalar evaluation).
    add("signal_lookup_1d",  {true, true, false, false, false, FIM});
    add("signal_lookup_2d",  {true, true, false, false, false, FIM});
    // Math.
    add("signal_gain",       {true, true, false, false, false, FIM});
    add("signal_sum",        {true, true, false, false, false, FIM});
    add("signal_product",    {true, true, false, false, false, FIM});
    add("signal_abs",        {true, true, false, false, false, FIM});
    add("signal_saturation", {true, true, false, false, true,  FIM});
    add("signal_math_fcn",   {true, true, false, false, false, FIM});
    add("signal_trig_fcn",   {true, true, false, false, false, FIM});
    add("signal_dead_zone",  {true, true, false, false, false, FIM});
    add("signal_relop",      {true, true, false, false, false, FIM});
    add("signal_logical",    {true, true, false, false, false, FIM});
    add("signal_compare_to_zero",
                             {true, true, false, false, false, FIM});
    add("signal_compare_to_constant",
                             {true, true, false, false, false, FIM});
    // Tier-H carve-out — Custom Block (MATLAB Function). The
    // `params.expression` string is parsed once at construction
    // (`MflowLinkSim::buildMatlabFcn`); the evaluator walks the
    // resulting tiny expression tree on every step. Supports the
    // arithmetic / math / trig functions documented in
    // `docs/mflowlink_blocks.md`. Variable names: `u` / `u1` …
    // `uN` for the N input ports, plus `t` for simulation time.
    add("signal_matlab_fcn", {true, true, false, false, false, FIM});
    // Signal routing.
    add("signal_mux",    {true, true, false, false, false, FIM});
    add("signal_demux",  {true, true, false, false, false, FIM});
    add("signal_switch", {true, true, false, false, true,  FIM});
    add("signal_multiport_switch",
                         {true, true, false, false, false, FIM});
    add("signal_merge",  {true, true, false, false, false, FIM});
    // Tier-H carve-out — virtual wires. Marked Composite so the
    // Flattener's existing "contracted away during lowering" path
    // applies; a dedicated `contractGotoFrom` pass actually rewires
    // edges across them. By the time block construction runs,
    // every goto and from has been removed from the graph.
    add("signal_goto", {true, true, true, false, false, FIM});
    add("signal_from", {true, true, true, false, false, FIM});
    // Tier E carve-out — hysteretic relay (the third zero-crossing
    // kind alongside Switch / Saturation in roadmap §7.3). One
    // discrete state slot for the latched on/off bit; ZC predicate
    // flips when the input crosses either threshold rail.
    add("signal_relay",  {true, true, false, false, true,  FIM});
    // Composite — handled by flattening, never reaches block construction.
    add("signal_subsystem", {true, true, true, false, false, FIM});
    add("signal_inport",    {true, true, true, false, false, FIM});
    add("signal_outport",   {true, true, true, false, false, FIM});
    // Tier F — gated composites. Flattening is identical to
    // `signal_subsystem`; the extra step is stamping every inlined
    // child block with the enable source named by
    // `data.enable_block` (a sibling block in the parent flow). The
    // runtime evaluator (`MflowLinkSim::evalAll`) holds outputs and
    // zeros derivatives while the gate is ≤ 0. Triggered / function-
    // call variants accept the same `data.enable_block` field today
    // and share the same gate semantics — proper edge-trigger /
    // call-event handling is Tier H.
    add("signal_enabled_subsystem",   {true, true, true, false, false, FIM});
    add("signal_triggered_subsystem", {true, true, true, false, false, FIM});

    // --- Reserved (Known, not yet Supported) --------------------------------
    // The remaining reserved kinds need prerequisites the runtime
    // doesn't have yet (vector signal type for bus creator/selector,
    // workspace var binding for from_workspace, an inline MATLAB
    // expression evaluator for matlab_fcn / custom, parent If /
    // SwitchCase subsystem containers for *_action, the goto / from
    // virtual-wire lowering pass, and the N-d generalisation after
    // lookup_1d / 2d are solid). See `docs/mflowlink_blocks.md`.
    for (const char *Name : {
             "signal_from_workspace",
             "signal_bus_creator", "signal_bus_selector",
             "signal_if_action", "signal_switch_case_action",
             "signal_lookup_nd",
             "signal_custom"}) { // NOTE: signal_custom remains
                                  // reserved — it needs a plugin
                                  // hook the runtime doesn't have
      KindInfo I;
      I.Known = true;
      I.Supported = false;
      T[Name] = I;
    }
    return T;
  }();
  auto It = Table.find(K);
  return It == Table.end() ? nullptr : &It->second;
}

//===----------------------------------------------------------------------===//
// Small numeric helpers over the raw-text param strings.
//===----------------------------------------------------------------------===//

// Degree of a comma-separated coefficient list ("1, 2, 1" → 2). An
// empty / single-term list is degree 0.
int polyDegree(const std::string &S) {
  int Terms = 0;
  size_t I = 0;
  while (I <= S.size()) {
    size_t J = S.find(',', I);
    std::string Tok = S.substr(I, J == std::string::npos ? std::string::npos
                                                         : J - I);
    // trim
    size_t A = Tok.find_first_not_of(" \t");
    if (A != std::string::npos) ++Terms;
    if (J == std::string::npos) break;
    I = J + 1;
  }
  return Terms > 0 ? Terms - 1 : 0;
}

// Row count of a MATLAB-ish matrix literal ("[0 1; -1 0]" → 2,
// "0" → 1). Counts `;` separators inside the outermost brackets.
int matrixRows(const std::string &Raw) {
  std::string S = Raw;
  size_t A = S.find_first_not_of(" \t");
  size_t B = S.find_last_not_of(" \t");
  if (A == std::string::npos) return 0;
  S = S.substr(A, B - A + 1);
  if (!S.empty() && S.front() == '[') S.erase(S.begin());
  if (!S.empty() && S.back() == ']') S.pop_back();
  if (S.find_first_not_of(" \t") == std::string::npos) return 0;
  int Rows = 1;
  for (char C : S)
    if (C == ';') ++Rows;
  return Rows;
}

double parseDoubleOr(const std::string *S, double Fallback) {
  if (!S) return Fallback;
  try {
    return std::stod(*S);
  } catch (...) {
    return Fallback;
  }
}

//===----------------------------------------------------------------------===//
// Subsystem flattening (§6.2).
//
// Recursively inline every `signal_subsystem` into its parent. The
// boundary tags — `signal_inport` / `signal_outport` — survive the
// recursion as single-in/single-out passthrough nodes, then a final
// contraction pass splices them out, leaving one flat block graph the
// runtime never sees a subsystem in.
//===----------------------------------------------------------------------===//

struct FNode {
  std::string Id;          // flat (prefixed) id
  const Node *Src;         // originating loader node
  // Tier-F enable-gate inheritance. Empty for a top-level block
  // (always enabled). For a block inlined from a
  // `signal_enabled_subsystem` / `signal_triggered_subsystem`, holds
  // the flat id of the source signal driving the subsystem's enable.
  std::string EnableStamp;
  // Tier-F carve-out — true when the stamp came from a
  // `signal_triggered_subsystem`, so the runtime treats the gate as
  // a rising-edge condition rather than a level condition.
  bool EdgeTriggered = false;
};

struct FEdge {
  std::string Id;
  std::string FromNode, FromPort;
  std::string ToNode, ToPort;
};

struct FlatGraph {
  std::vector<FNode> Nodes;
  std::vector<FEdge> Edges;
};

class Flattener {
public:
  Flattener(const FlowDoc &Doc, DiagnosticEngine &Diag)
      : Doc_(Doc), Diag_(Diag) {}

  std::optional<FlatGraph> run(const Flow &Entry) {
    std::vector<std::string> Stack{Entry.Id};
    auto G = expand(Entry, "", Stack, /*EnableStamp=*/"",
                    /*EdgeTriggered=*/false);
    if (!G) return std::nullopt;
    if (!contractBoundaries(*G)) return std::nullopt;
    if (!contractGotoFrom(*G)) return std::nullopt;
    return G;
  }

private:
  const FlowDoc &Doc_;
  DiagnosticEngine &Diag_;
  int SynthEdge_ = 0;

  // Compute the effective enable stamp for an inherited gate +
  // optional override. Empty if neither is set. When BOTH are set
  // (a per-block `data.enable_block` inside an already-gated
  // subsystem), we keep the inherited gate — the subsystem-level
  // enable is the outer guard, and an inner override should be
  // wired through the diagram, not stacked here. A real
  // multi-gate composite is a Tier-H follow-up.
  static std::string composeEnable(const std::string &Inherited,
                                   const std::string &Local,
                                   const std::string &Prefix) {
    if (!Inherited.empty()) return Inherited;
    if (Local.empty()) return std::string{};
    return Prefix + Local;
  }

  const Flow *findFlowById(const std::string &Id) const {
    for (auto &F : Doc_.Flows)
      if (F.Id == Id) return &F;
    return nullptr;
  }

  // Which subsystem external port a boundary node binds to: an
  // explicit `data.port`, else the node's own id.
  static std::string bindingPort(const Node &N) {
    if (auto *P = N.getData("port")) return *P;
    return N.Id;
  }

  // Recursively expand `F`. `Prefix` namespaces every emitted id.
  // `Stack` carries the flow ids currently being expanded — a flow
  // id reappearing is a subsystem cycle. `EnableStamp` (Tier F) is
  // the inherited conditional-subsystem gate: every leaf inlined
  // here picks it up unless we descend into a nested enabled-
  // subsystem (handled by `composeEnable`). `EdgeTriggered` flags
  // the stamp as a rising-edge condition (from a
  // `signal_triggered_subsystem`) rather than a level condition.
  std::optional<FlatGraph> expand(const Flow &F, const std::string &Prefix,
                                  std::vector<std::string> &Stack,
                                  const std::string &EnableStamp,
                                  bool EdgeTriggered) {
    FlatGraph G;
    // subsystem node id → { bindingPort → flat boundary node id }
    std::unordered_map<std::string, std::map<std::string, std::string>> InMap;
    std::unordered_map<std::string, std::map<std::string, std::string>> OutMap;
    std::unordered_set<std::string> SubsysIds;

    for (auto &N : F.Nodes) {
      const KindInfo *KI = lookupKind(N.Kind);
      const bool IsSubsystem =
          KI && KI->Composite &&
          (N.Kind == "signal_subsystem" ||
           N.Kind == "signal_enabled_subsystem" ||
           N.Kind == "signal_triggered_subsystem");
      if (IsSubsystem) {
        SubsysIds.insert(N.Id);
        const std::string *FlowId = N.getData("flow_id");
        if (!FlowId || FlowId->empty()) {
          Diag_.error(N.Loc, N.Kind + " \"" + N.Id +
                                 "\" is missing data.flow_id");
          return std::nullopt;
        }
        const Flow *Child = findFlowById(*FlowId);
        if (!Child) {
          Diag_.error(N.Loc, N.Kind + " \"" + N.Id +
                                 "\" references unknown flow id \"" + *FlowId +
                                 "\"");
          return std::nullopt;
        }
        if (std::find(Stack.begin(), Stack.end(), *FlowId) != Stack.end()) {
          Diag_.error(N.Loc, "subsystem cycle: flow \"" + *FlowId +
                                 "\" is already being expanded (a subsystem "
                                 "may not reference an ancestor flow)");
          return std::nullopt;
        }
        // Inherit the parent's gate (if any) plus the subsystem's
        // own `data.enable_block` override. The edge-triggered flag
        // is set only when descending into a triggered subsystem
        // that introduces its own gate — nested enabled subsystems
        // under a triggered one inherit the level-gated semantic
        // of their immediate parent (Tier H may revisit this).
        std::string ChildEnable = EnableStamp;
        bool ChildEdge = EdgeTriggered;
        if (N.Kind == "signal_enabled_subsystem" ||
            N.Kind == "signal_triggered_subsystem") {
          if (auto *EB = N.getData("enable_block")) {
            std::string Composed =
                composeEnable(EnableStamp, *EB, Prefix);
            if (Composed != EnableStamp) {
              ChildEnable = Composed;
              ChildEdge = (N.Kind == "signal_triggered_subsystem");
            }
          }
        }
        Stack.push_back(*FlowId);
        auto ChildG = expand(*Child, Prefix + N.Id + "/", Stack,
                             ChildEnable, ChildEdge);
        Stack.pop_back();
        if (!ChildG) return std::nullopt;
        for (auto &CN : ChildG->Nodes) {
          if (CN.Src->Kind == "signal_inport")
            InMap[N.Id][bindingPort(*CN.Src)] = CN.Id;
          else if (CN.Src->Kind == "signal_outport")
            OutMap[N.Id][bindingPort(*CN.Src)] = CN.Id;
          G.Nodes.push_back(CN);
        }
        for (auto &CE : ChildG->Edges) G.Edges.push_back(CE);
        continue;
      }
      // Leaf block, or an inport/outport passthrough — keep it and
      // stamp the active enable inheritance. A leaf's own
      // `data.enable_block` is honoured directly when no outer
      // gate is in force; `composeEnable` resolves the conflict.
      std::string LeafEnable = EnableStamp;
      bool LeafEdge = EdgeTriggered;
      if (LeafEnable.empty())
        if (auto *EB = N.getData("enable_block")) {
          LeafEnable = composeEnable("", *EB, Prefix);
          // A leaf-level `enable_block` is always level-gated; the
          // edge form only comes from a triggered subsystem.
          LeafEdge = false;
        }
      G.Nodes.push_back({Prefix + N.Id, &N, LeafEnable, LeafEdge});
    }

    // Rewire this flow's edges, redirecting through subsystem boundaries.
    for (auto &E : F.Edges) {
      FEdge FE;
      FE.Id = Prefix + E.Id;
      FE.FromPort = E.From.Port;
      FE.ToPort = E.To.Port;

      if (SubsysIds.count(E.From.Node)) {
        auto &M = OutMap[E.From.Node];
        auto It = M.find(E.From.Port);
        if (It == M.end()) {
          Diag_.error(E.Loc, "edge leaves subsystem \"" + E.From.Node +
                                 "\" via port \"" + E.From.Port +
                                 "\" but no signal_outport binds that port");
          return std::nullopt;
        }
        FE.FromNode = It->second;
        FE.FromPort = "out";
      } else {
        FE.FromNode = Prefix + E.From.Node;
      }

      if (SubsysIds.count(E.To.Node)) {
        auto &M = InMap[E.To.Node];
        auto It = M.find(E.To.Port);
        if (It == M.end()) {
          Diag_.error(E.Loc, "edge enters subsystem \"" + E.To.Node +
                                 "\" via port \"" + E.To.Port +
                                 "\" but no signal_inport binds that port");
          return std::nullopt;
        }
        FE.ToNode = It->second;
        FE.ToPort = "in";
      } else {
        FE.ToNode = Prefix + E.To.Node;
      }
      G.Edges.push_back(std::move(FE));
    }
    return G;
  }

  // Splice out every `signal_inport` / `signal_outport` passthrough:
  // each in-edge × each out-edge becomes one direct edge, then the
  // boundary node and its incident edges are dropped (§6.2).
  bool contractBoundaries(FlatGraph &G) {
    auto isBoundary = [](const FNode &N) {
      return N.Src->Kind == "signal_inport" || N.Src->Kind == "signal_outport";
    };
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (size_t I = 0; I < G.Nodes.size(); ++I) {
        if (!isBoundary(G.Nodes[I])) continue;
        const std::string BId = G.Nodes[I].Id;
        std::vector<FEdge> In, Out, Rest;
        for (auto &E : G.Edges) {
          if (E.ToNode == BId)        In.push_back(E);
          else if (E.FromNode == BId) Out.push_back(E);
          else                        Rest.push_back(E);
        }
        for (auto &IE : In)
          for (auto &OE : Out)
            Rest.push_back({"_splice" + std::to_string(SynthEdge_++),
                            IE.FromNode, IE.FromPort, OE.ToNode, OE.ToPort});
        G.Edges = std::move(Rest);
        G.Nodes.erase(G.Nodes.begin() + I);
        Changed = true;
        break;
      }
    }
    return true;
  }

  //===-----------------------------------------------------------===//
  // Tier-H carve-out — virtual wires (`signal_goto` / `signal_from`).
  //
  // Each `signal_goto` is a sink: it has a single incoming edge
  // carrying the value to broadcast, and a `data.tag` naming the
  // broadcast channel. Each `signal_from` is a source: no incoming
  // edge, a `data.tag` naming the channel it subscribes to. After
  // this pass, every outgoing edge of every `signal_from` is
  // rewired so its source is the block driving the matching
  // `signal_goto`, and both kinds are removed from the graph. Net
  // effect: the goto/from pair act like a long-distance wire that
  // never appears in the runtime IR.
  //===-----------------------------------------------------------===//
  bool contractGotoFrom(FlatGraph &G) {
    auto dataTag = [](const Node &N) -> std::string {
      if (auto *T = N.getData("tag")) return *T;
      return std::string{};
    };
    // tag → (source flat block id, source port). Filled from each
    // signal_goto's single incoming edge.
    std::unordered_map<std::string, std::pair<std::string, std::string>>
        TagSrc;
    std::unordered_set<std::string> Drop;
    for (auto &N : G.Nodes) {
      if (N.Src->Kind == "signal_goto") {
        std::string Tag = dataTag(*N.Src);
        if (Tag.empty()) {
          Diag_.error(N.Src->Loc,
                      "signal_goto \"" + N.Id + "\" missing data.tag");
          return false;
        }
        const FEdge *Inc = nullptr;
        int InCount = 0;
        for (auto &E : G.Edges) {
          if (E.ToNode == N.Id) { Inc = &E; ++InCount; }
        }
        if (InCount == 0) {
          Diag_.error(N.Src->Loc, "signal_goto \"" + N.Id +
                                      "\" (tag \"" + Tag +
                                      "\") has no incoming signal");
          return false;
        }
        if (InCount > 1) {
          Diag_.error(N.Src->Loc, "signal_goto \"" + N.Id +
                                      "\" has " + std::to_string(InCount) +
                                      " incoming edges; expected exactly 1");
          return false;
        }
        if (TagSrc.count(Tag)) {
          Diag_.error(N.Src->Loc, "duplicate signal_goto for tag \"" +
                                      Tag + "\"");
          return false;
        }
        TagSrc[Tag] = {Inc->FromNode, Inc->FromPort};
        Drop.insert(N.Id);
      } else if (N.Src->Kind == "signal_from") {
        Drop.insert(N.Id);
      }
    }
    if (Drop.empty()) return true;

    // Rewire edges. An edge from a signal_from gets its source
    // replaced by the tag's stored producer. Edges into a goto or
    // edges incident to a from on the input side are dropped —
    // gotos consume their input, froms have no input.
    auto lookupKindFor = [&](const std::string &Id) -> const std::string * {
      for (auto &N : G.Nodes)
        if (N.Id == Id) return &N.Src->Kind;
      return nullptr;
    };
    auto lookupNodeFor = [&](const std::string &Id) -> const Node * {
      for (auto &N : G.Nodes)
        if (N.Id == Id) return N.Src;
      return nullptr;
    };
    std::vector<FEdge> Kept;
    Kept.reserve(G.Edges.size());
    for (auto &E : G.Edges) {
      if (Drop.count(E.ToNode)) {
        // Edge into goto = consumed. Edge into from = malformed
        // (from has no input port); the loader's port validation
        // would have caught it. Either way, drop.
        continue;
      }
      if (Drop.count(E.FromNode)) {
        const std::string *K = lookupKindFor(E.FromNode);
        if (K && *K == "signal_from") {
          const Node *FromN = lookupNodeFor(E.FromNode);
          std::string Tag = FromN ? dataTag(*FromN) : std::string{};
          auto It = TagSrc.find(Tag);
          if (It == TagSrc.end()) {
            Diag_.error(FromN ? FromN->Loc : SourceLocation{},
                        "signal_from \"" + E.FromNode +
                            "\" references tag \"" + Tag +
                            "\" which no signal_goto provides");
            return false;
          }
          FEdge Rewired = E;
          Rewired.FromNode = It->second.first;
          Rewired.FromPort = It->second.second;
          Kept.push_back(std::move(Rewired));
          continue;
        }
        // Edge from goto: gotos are sinks, so an outgoing edge is
        // malformed. Drop defensively.
        continue;
      }
      Kept.push_back(E);
    }
    G.Edges = std::move(Kept);
    G.Nodes.erase(
        std::remove_if(G.Nodes.begin(), G.Nodes.end(),
                       [&](const FNode &N) { return Drop.count(N.Id) > 0; }),
        G.Nodes.end());
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public helpers
//===----------------------------------------------------------------------===//

const char *sampleTimeClassName(SampleTimeClass C) {
  switch (C) {
  case SampleTimeClass::Continuous:   return "continuous";
  case SampleTimeClass::Discrete:     return "discrete";
  case SampleTimeClass::Constant:     return "constant";
  case SampleTimeClass::FixedInMinor: return "fixed_in_minor";
  }
  return "fixed_in_minor";
}

const MflBlock *MflowLinkModel::findBlock(const std::string &Id) const {
  for (auto &B : Blocks)
    if (B.Id == Id) return &B;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// lowerSignalFlow
//===----------------------------------------------------------------------===//

std::optional<MflowLinkModel> lowerSignalFlow(const FlowDoc &Doc,
                                              DiagnosticEngine &Diag) {
  if (!Doc.isSignalFlow()) {
    SourceLocation Loc;
    Loc.File = Doc.File;
    Diag.error(Loc, "lowerSignalFlow: document is not a signal-flow .mflow "
                    "(settings.kind != \"signal_flow\")");
    return std::nullopt;
  }

  // Resolve the entry flow. An explicit `entry` wins; otherwise a
  // single-flow document uses its only flow.
  const Flow *Entry = Doc.entryFlow();
  if (!Entry && Doc.Entry.empty() && Doc.Flows.size() == 1)
    Entry = &Doc.Flows.front();
  if (!Entry) {
    SourceLocation Loc;
    Loc.File = Doc.File;
    Diag.error(Loc, "signal-flow document has no resolvable entry flow");
    return std::nullopt;
  }

  // Flatten subsystems into one block graph.
  Flattener FL(Doc, Diag);
  auto Flat = FL.run(*Entry);
  if (!Flat) return std::nullopt;

  MflowLinkModel M;
  M.EntryName = Entry->Name;
  M.Solver = Doc.Settings.Solver.value_or(SolverConfig{});
  M.Snapshot = Doc.Settings.Snapshot.value_or(SnapshotConfig{});

  // Build the block list.
  for (auto &FN : Flat->Nodes) {
    const Node &N = *FN.Src;
    const KindInfo *KI = lookupKind(N.Kind);
    if (!KI || !KI->Known) {
      Diag.error(N.Loc, "unknown signal-flow block kind \"" + N.Kind + "\"");
      return std::nullopt;
    }
    if (KI->Composite) {
      // signal_subsystem is replaced during flattening; inport/outport
      // are contracted away. Reaching here means a malformed graph.
      Diag.error(N.Loc, "composite block \"" + N.Kind +
                            "\" survived flattening — malformed subsystem");
      return std::nullopt;
    }
    if (!KI->Supported) {
      Diag.error(N.Loc, "signal-flow block kind \"" + N.Kind +
                            "\" is reserved but its evaluator has not shipped "
                            "yet — not supported by lowering");
      return std::nullopt;
    }

    MflBlock B;
    B.Id = FN.Id;
    B.Kind = N.Kind;
    B.Params = N.Params;
    B.Loc = N.Loc;
    B.SampleClass = KI->Sample;
    B.EnableSource = FN.EnableStamp;
    B.EnableEdgeTriggered = FN.EdgeTriggered;
    if (auto *L = N.getData("log_signal"))
      B.LogSignal = (*L == "true");

    // State counts + loop-breaker classification.
    B.IsLoopBreaker = KI->LoopBreakerAlways;
    if (N.Kind == "signal_integrator") {
      B.ContStateCount = 1;
    } else if (N.Kind == "signal_transfer_fcn") {
      int DenDeg = polyDegree(N.getParam("den") ? *N.getParam("den") : "1");
      int NumDeg = polyDegree(N.getParam("num") ? *N.getParam("num") : "1");
      B.ContStateCount = DenDeg;
      // A strictly-proper transfer function has no direct feedthrough,
      // so it breaks an algebraic loop just like an integrator does.
      if (NumDeg < DenDeg) B.IsLoopBreaker = true;
    } else if (N.Kind == "signal_state_space") {
      B.ContStateCount = matrixRows(N.getParam("A") ? *N.getParam("A") : "");
      const std::string *D = N.getParam("D");
      if (!D || *D == "0" || D->empty()) B.IsLoopBreaker = true;
    } else if (N.Kind == "signal_relay") {
      // One discrete bit (on / off). The relay is a fixed-in-minor
      // block — its output is read every step, but state transitions
      // only at major-step boundaries (the evaluator skips updates
      // when `Deriv != nullptr`, i.e. inside RK4 substeps).
      B.DiscStateCount = 1;
    } else if (N.Kind == "signal_unit_delay" || N.Kind == "signal_zoh" ||
               N.Kind == "signal_discrete_integrator" ||
               N.Kind == "signal_discrete_filter" ||
               N.Kind == "signal_rate_transition") {
      // Discrete period: `params.sampleTime`, else numeric
      // `data.sample_time`, else 1 s. Every discrete block keys
      // its NextFire_ on this period; the evaluator-specific state
      // shape (single-bit latch for ZOH / unit_delay, accumulator
      // for discrete_integrator, IIR taps for discrete_filter, etc.)
      // is decided in MflowLinkSim. For Tier-H pass 3 we keep one
      // scalar slot — vector filters land alongside bus signals.
      B.DiscStateCount = 1;
      if (N.Kind == "signal_discrete_filter") {
        int DenDeg = polyDegree(N.getParam("den") ? *N.getParam("den") : "1");
        B.DiscStateCount = std::max(1, DenDeg);
      }
      double Period = parseDoubleOr(N.getParam("sampleTime"), -1.0);
      if (Period < 0.0) {
        if (auto *ST = N.getData("sample_time"))
          Period = parseDoubleOr(ST, 1.0);
        else
          Period = 1.0;
      }
      B.SamplePeriod = Period;
    } else if (N.Kind == "signal_zero_pole") {
      // Convert zeros/poles to num/den polynomial degrees so the
      // existing transfer-fcn machinery (state count + loop breaker
      // classification) applies as-is. The evaluator does the actual
      // ZPK → coefficient expansion at construction time.
      int Z = 0, P = 0;
      if (auto *S = N.getParam("zeros")) Z = polyDegree(*S) + 1;
      if (auto *S = N.getParam("poles")) P = polyDegree(*S) + 1;
      B.ContStateCount = P > 0 ? P : 0;
      if (Z < P) B.IsLoopBreaker = true;
    } else if (N.Kind == "signal_transport_delay") {
      // Pure time delay needs a history buffer. Sample class is
      // Continuous because the runtime reads the delayed value at
      // every step (interpolating the buffer); the buffer itself
      // lives in a per-block stash inside MflowLinkSim, not in Y_.
      B.IsLoopBreaker = true; // delayed-by-anything-positive ⇒ no feedthrough
    }

    M.ContStateCount += B.ContStateCount;
    M.DiscStateCount += B.DiscStateCount;
    if (KI->ZeroCrossing)
      M.ZeroCrossings.push_back({B.Id, B.Kind});
    M.Blocks.push_back(std::move(B));
  }

  // Build the edge list.
  for (auto &FE : Flat->Edges)
    M.Edges.push_back({FE.Id, FE.FromNode, FE.FromPort, FE.ToNode, FE.ToPort});

  // Validate every `EnableSource` resolves to a block we know about.
  // A typo here would silently disable a subtree at runtime — far
  // better to fail cleanly at lower time with the block id quoted.
  for (auto &B : M.Blocks) {
    if (B.EnableSource.empty()) continue;
    bool Found = false;
    for (auto &Other : M.Blocks)
      if (Other.Id == B.EnableSource) { Found = true; break; }
    if (!Found) {
      Diag.error(B.Loc, "block \"" + B.Id +
                            "\" references unknown enable block \"" +
                            B.EnableSource + "\"");
      return std::nullopt;
    }
  }

  // Tier-H carve-out — validate every `signal_matlab_fcn`'s
  // `params.expression` parses cleanly. The user gets a sourced
  // diagnostic at lower time; the runtime cache then trusts the
  // parse will succeed.
  for (auto &B : M.Blocks) {
    if (B.Kind != "signal_matlab_fcn") continue;
    auto It = B.Params.find("expression");
    if (It == B.Params.end() || It->second.empty()) {
      Diag.error(B.Loc, "signal_matlab_fcn \"" + B.Id +
                            "\" missing data.params.expression");
      return std::nullopt;
    }
    std::string Err = validateMatlabFcnExpression(It->second);
    if (!Err.empty()) {
      Diag.error(B.Loc, "signal_matlab_fcn \"" + B.Id +
                            "\": " + Err);
      return std::nullopt;
    }
  }

  //===--------------------------------------------------------------------===//
  // Execution-order sort (§6.3).
  //
  // Topological sort over data edges. A loop-breaker's *outgoing*
  // edges are dropped — its output this step comes from state, not
  // from this step's input. Whatever remains acyclic is the
  // direct-feedthrough order; anything still cyclic is an algebraic
  // loop (§6.4).
  //===--------------------------------------------------------------------===//
  std::unordered_map<std::string, size_t> IndexOf;
  for (size_t I = 0; I < M.Blocks.size(); ++I)
    IndexOf[M.Blocks[I].Id] = I;

  const size_t N = M.Blocks.size();
  std::vector<int> InDegree(N, 0);
  std::vector<std::vector<size_t>> Succ(N);
  for (auto &E : M.Edges) {
    auto F = IndexOf.find(E.FromBlock);
    auto T = IndexOf.find(E.ToBlock);
    if (F == IndexOf.end() || T == IndexOf.end())
      continue; // dangling — already diagnosed by the loader
    if (M.Blocks[F->second].IsLoopBreaker)
      continue; // drop the loop-breaker's outgoing edge
    Succ[F->second].push_back(T->second);
    ++InDegree[T->second];
  }

  // Kahn's algorithm, lowest block index first for a deterministic order.
  std::set<size_t> Ready;
  for (size_t I = 0; I < N; ++I)
    if (InDegree[I] == 0) Ready.insert(I);
  while (!Ready.empty()) {
    size_t I = *Ready.begin();
    Ready.erase(Ready.begin());
    M.ExecOrder.push_back(I);
    for (size_t S : Succ[I])
      if (--InDegree[S] == 0) Ready.insert(S);
  }

  if (M.ExecOrder.size() != N) {
    // The blocks left unscheduled are the algebraic loop *plus*
    // anything downstream of it. Peel nodes with no successor still
    // in the set until only the genuine cycles remain — so the
    // diagnostic names the blocks the user actually has to break.
    std::unordered_set<size_t> U;
    for (size_t I = 0; I < N; ++I)
      if (InDegree[I] > 0) U.insert(I);
    bool Peeled = true;
    while (Peeled) {
      Peeled = false;
      for (size_t I : std::vector<size_t>(U.begin(), U.end())) {
        bool HasSuccInU = false;
        for (size_t S : Succ[I])
          if (U.count(S)) { HasSuccInU = true; break; }
        if (!HasSuccInU) {
          U.erase(I);
          Peeled = true;
        }
      }
    }
    std::vector<size_t> OnLoop(U.begin(), U.end());
    std::sort(OnLoop.begin(), OnLoop.end());
    std::string List;
    for (size_t K = 0; K < OnLoop.size(); ++K) {
      if (K) List += ", ";
      List += "\"" + M.Blocks[OnLoop[K]].Id + "\"";
    }
    Diag.error(M.Blocks[OnLoop.front()].Loc,
               "algebraic loop in signal-flow model: blocks " + List +
                   " form a direct-feedthrough cycle with no loop-breaker "
                   "(insert an Integrator, Unit Delay, or ZOH to break it)");
    return std::nullopt;
  }

  return M;
}

//===----------------------------------------------------------------------===//
// dumpMflowLinkModel — `matlabc -simulate --dry-run`
//===----------------------------------------------------------------------===//

void dumpMflowLinkModel(std::ostream &OS, const MflowLinkModel &M) {
  OS << "MflowLinkModel entry=" << M.EntryName << " blocks=" << M.Blocks.size()
     << " edges=" << M.Edges.size() << "\n";
  OS << "  solver type=" << M.Solver.Type << " algorithm=" << M.Solver.Algorithm
     << " startTime=" << M.Solver.StartTime << " stopTime=" << M.Solver.StopTime
     << "\n";
  OS << "  states continuous=" << M.ContStateCount
     << " discrete=" << M.DiscStateCount << "\n";
  OS << "  exec-order:\n";
  for (size_t Pos = 0; Pos < M.ExecOrder.size(); ++Pos) {
    const MflBlock &B = M.Blocks[M.ExecOrder[Pos]];
    OS << "    " << Pos << " " << B.Id << " kind=" << B.Kind
       << " sample=" << sampleTimeClassName(B.SampleClass);
    if (B.SampleClass == SampleTimeClass::Discrete)
      OS << " period=" << B.SamplePeriod;
    if (B.ContStateCount) OS << " cont-states=" << B.ContStateCount;
    if (B.DiscStateCount) OS << " disc-states=" << B.DiscStateCount;
    if (B.IsLoopBreaker) OS << " loop-breaker";
    if (B.LogSignal) OS << " log";
    if (!B.EnableSource.empty()) {
      OS << " enable=" << B.EnableSource;
      if (B.EnableEdgeTriggered) OS << " (rising-edge)";
    }
    OS << "\n";
  }
  OS << "  zero-crossings:";
  if (M.ZeroCrossings.empty()) {
    OS << " none\n";
  } else {
    OS << "\n";
    for (auto &Z : M.ZeroCrossings)
      OS << "    " << Z.BlockId << " (" << Z.Kind << ")\n";
  }
  OS << "  edges:\n";
  for (auto &E : M.Edges)
    OS << "    " << E.Id << " " << E.FromBlock << ":" << E.FromPort << " -> "
       << E.ToBlock << ":" << E.ToPort << "\n";
}

} // namespace matlab::flowchart
