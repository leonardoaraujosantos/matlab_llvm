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

const std::map<std::string, KindInfo> &kindTable() {
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
    // Parallel PID with derivative filter. Direct-feedthrough
    // (C(∞) = Kp + Kd·N is finite), so it is NOT a loop breaker.
    add("signal_pid",          {true, true, false, false, false, CONT});
    // Control (#343) — static state-feedback gain: u = -K·x (LQR / pole
    // placement). Direct-feedthrough matrix-vector product, not a loop breaker.
    add("signal_lqr",          {true, true, false, false, false, FIM});
    // Deep Learning (#343) — feedforward MLP inference in a loop. One hidden
    // layer: y = W2·act(W1·x + b1) + b2. Stateless, direct-feedthrough.
    add("signal_dnn_predict",  {true, true, false, false, false, FIM});
    // Reinforcement Learning (#343) — trained deterministic policy in the loop.
    // An MLP maps state → action; discrete picks argmax (action index),
    // continuous emits actionScale·tanh(raw). Stateless, direct-feedthrough.
    add("signal_rl_agent",     {true, true, false, false, false, FIM});
    add("signal_transport_delay",
                               {true, true, false, true,  false, CONT});
    // Discrete.
    add("signal_unit_delay", {true, true, false, true, false, DISC});
    add("signal_zoh",        {true, true, false, true, false, DISC});
    add("signal_discrete_integrator",
                             {true, true, false, true, false, DISC});
    add("signal_discrete_filter",
                             {true, true, false, true, false, DISC});
    // DSP (#343) — biquad / second-order section. A 2nd-order IIR routed
    // through the same discrete-filter difference-engine, with named SOS
    // coefficients (b0..b2 / a0..a2) instead of num/den polynomials.
    add("signal_biquad",     {true, true, false, true, false, DISC});
    // DSP (#343) — streaming first-order filters, also discrete_filter presets:
    //   signal_lowpass   one-pole EMA   H(z) = α / (1 - (1-α)z⁻¹)
    //   signal_highpass  one-pole HP    H(z) = α(1 - z⁻¹) / (1 - α z⁻¹)
    //   signal_dcblock   DC blocker     H(z) = (1 - z⁻¹) / (1 - r z⁻¹)
    add("signal_lowpass",    {true, true, false, true, false, DISC});
    add("signal_highpass",   {true, true, false, true, false, DISC});
    add("signal_dcblock",    {true, true, false, true, false, DISC});
    add("signal_rate_transition",
                             {true, true, false, true, false, DISC});
    // HDL / digital sequential elements (#343) — clocked registers driven
    // by an external `clk` rising edge (not a fixed sample rate), so they're
    // FixedInMinor + always loop-breakers (output comes from held state).
    // Maps to the synchronous emit-{systemverilog,verilog,cocotb} lane;
    // emit lowering is a follow-up. Mux/Demux + logic gates already ship as
    // signal_mux/demux/logical/multiport_switch.
    add("signal_dff",        {true, true, false, true, false, FIM});
    add("signal_tff",        {true, true, false, true, false, FIM});
    add("signal_counter",    {true, true, false, true, false, FIM});
    // JK / SR flip-flops — same clocked single-latch family as D/T: edge-
    // triggered, loop-breaking, held in DigitalLatch_ outside the state vectors.
    add("signal_jkff",       {true, true, false, true, false, FIM});
    add("signal_srff",       {true, true, false, true, false, FIM});
    // HDL memory (#343) — array-state blocks. Shift register and RAM update on
    // a `clk` posedge (loop-breaking, held array); ROM is a combinational
    // address→value lookup (stateless). Array state lives in HdlMem_.
    add("signal_shift_register", {true, true, false, true, false, FIM});
    add("signal_ram",        {true, true, false, true,  false, FIM});
    add("signal_rom",        {true, true, false, false, false, FIM});
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
    // Communications (#343) — first toolbox-domain library block via the
    // mflow-toolbox-library-blocks recipe. AWGN channel: direct-feedthrough
    // additive noise, stateless aside from the per-block RNG seed.
    add("signal_awgn",       {true, true, false, false, false, FIM});
    // Communications (#343) — error-rate (BER) sink. Output is the running
    // mismatch ratio between two inputs; the accumulation is stateful (carried
    // across steps), so it breaks algebraic loops like the clocked registers.
    add("signal_error_rate", {true, true, false, true,  false, FIM});
    // Communications (#343) — PSK / QAM modulators & demodulators. Stateless,
    // direct-feedthrough constellation map (symbol → I/Q vector, width 2) and
    // demap (I/Q vector → nearest symbol index). Not loop breakers.
    add("signal_psk_mod",    {true, true, false, false, false, FIM});
    add("signal_psk_demod",  {true, true, false, false, false, FIM});
    add("signal_qam_mod",    {true, true, false, false, false, FIM});
    add("signal_qam_demod",  {true, true, false, false, false, FIM});
    // Computer Vision / Image Processing (#343, mflow-2d-image-signals) —
    // grayscale image blocks over the flattened row-major 2-D signal. All
    // stateless, direct-feedthrough. image_source defines its shape from
    // rows/cols; image_filter and threshold preserve the input image shape.
    add("signal_image_source", {true, true, false, false, false, FIM});
    add("signal_image_filter", {true, true, false, false, false, FIM});
    add("signal_threshold",    {true, true, false, false, false, FIM});
    // Statistics (#343) — streaming mean/variance/std over the input via an
    // online Welford accumulator. Stateful (carries across steps), so it
    // breaks algebraic loops; beats a MATLAB Function block, which can't hold
    // persistent state in the flow today.
    add("signal_running_stats", {true, true, false, true, false, FIM});
    // Sensor Fusion (#343) — discrete Kalman filter. Outputs the N-vector state
    // estimate (N = rows of A); stateful predict/update each major step, so it
    // breaks algebraic loops (the estimate is prior-derived, not feedthrough).
    add("signal_kalman",     {true, true, false, true,  false, FIM});
    // DSP (#343) — frame transforms over a vector signal. Stateless,
    // direct-feedthrough (output is this step's frame), so not loop breakers.
    // signal_fft maps a real N-frame → complex [Re;Im] (width 2N); signal_ifft
    // inverts it; signal_window applies a Hann/Hamming/Blackman taper.
    add("signal_fft",        {true, true, false, false, false, FIM});
    add("signal_ifft",       {true, true, false, false, false, FIM});
    add("signal_window",     {true, true, false, false, false, FIM});
    // signal_spectrum maps a real N-frame → its power spectrum |X[k]|² (width N).
    add("signal_spectrum",   {true, true, false, false, false, FIM});
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
    // §17.5 #9 — shape-aware reshape. Takes one input of total
    // element count R·C and re-emits it as a matrix-shaped signal
    // (rows × cols). Pure relabeling: flat storage stays the
    // same, downstream broadcast operators read the (R, C) shape
    // off the block's OutRows / OutCols. Total element count
    // must match the input width — enforced by the width-
    // inference pass.
    add("signal_reshape", {true, true, false, false, false, FIM});
    add("signal_switch", {true, true, false, false, true,  FIM});
    add("signal_multiport_switch",
                         {true, true, false, false, false, FIM});
    add("signal_merge",  {true, true, false, false, false, FIM});
    // §17.5 #1 — bus signals. The creator packs scalar inputs into
    // a named-field vector; the selector projects out one named
    // field. Both reuse the OutWidth + VecOut_ runtime path.
    add("signal_bus_creator",  {true, true, false, false, false, FIM});
    add("signal_bus_selector", {true, true, false, false, false, FIM});
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
    // MPC Toolbox Tier-3 §4.5 — MpcMove block.  Carries a static-
    // gain MPC approximation in the simulator (the QP-solving form
    // is a Tier-3b follow-up that needs runtime_mpc.cpp linked into
    // MatlabFlowchart).  Block parameters: `gain` (the static fb
    // gain), `r_default` (idle reference).  Two input ports (ym, r),
    // one output port (u).
    add("signal_mpc_move", {true, true, false, false, false, FIM});
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
  return Table;
}

const KindInfo *lookupKind(const std::string &K) {
  const auto &Table = kindTable();
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
  // Item-3 — mask param bindings inherited from the enclosing
  // `signal_subsystem` (if any). Empty for top-level blocks. The
  // MflBlock-construction step substitutes `${name}` placeholders
  // in this leaf's params with the bound values, so a library
  // block parameterised as `den: "1, ${tau}"` resolves to e.g.
  // `den: "1, 0.5"` once cloned under a host with `tau: 0.5`.
  std::map<std::string, std::string> MaskBindings;
  // §17.5 #7 — per-flow solver overrides effective for this leaf.
  // NaN when the enclosing flow inherits the global solver.
  double MaxStepOverride = std::numeric_limits<double>::quiet_NaN();
  double RelTolOverride  = std::numeric_limits<double>::quiet_NaN();
  double AbsTolOverride  = std::numeric_limits<double>::quiet_NaN();
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
    // §17.5 #7 — start with the global solver effective for the
    // entry flow (overridden by Entry.Solver if it carries one).
    SolverConfig InitSolver = Doc_.Settings.Solver.value_or(SolverConfig{});
    if (Entry.Solver) {
      // Apply per-flow overrides to the relevant fields.
      InitSolver.MaxStep = Entry.Solver->MaxStep;
      InitSolver.MinStep = Entry.Solver->MinStep;
      InitSolver.RelTol  = Entry.Solver->RelTol;
      InitSolver.AbsTol  = Entry.Solver->AbsTol;
    }
    auto G = expand(Entry, "", Stack, /*EnableStamp=*/"",
                    /*EdgeTriggered=*/false,
                    /*MaskBindings=*/{},
                    InitSolver);
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
                                  bool EdgeTriggered,
                                  std::map<std::string, std::string>
                                      MaskBindings,
                                  const SolverConfig &Solver) {
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
        // Item-3 — merge any subsystem-level `data.mask` bindings
        // into the child scope. The host's `data.mask.<param>`
        // shadows any inherited binding of the same name (inner
        // mask wins), so a library template can be nested under
        // another mask and still re-bind its own params.
        std::map<std::string, std::string> ChildMask = MaskBindings;
        if (auto *MaskParams = N.getData("mask_params")) {
          // The IDE serialises mask bindings as `data.mask_params`
          // — a flat string of `key=value, …` pairs. Parsing is
          // lenient: whitespace tolerated, malformed entries
          // skipped.
          std::string S = *MaskParams;
          size_t I = 0;
          while (I < S.size()) {
            size_t Eq = S.find('=', I);
            if (Eq == std::string::npos) break;
            size_t Comma = S.find(',', Eq + 1);
            if (Comma == std::string::npos) Comma = S.size();
            std::string K = S.substr(I, Eq - I);
            std::string V = S.substr(Eq + 1, Comma - Eq - 1);
            auto trim = [](std::string &T) {
              size_t A = T.find_first_not_of(" \t");
              size_t B = T.find_last_not_of(" \t");
              if (A == std::string::npos) { T.clear(); return; }
              T = T.substr(A, B - A + 1);
            };
            trim(K);
            trim(V);
            if (!K.empty()) ChildMask[K] = V;
            I = Comma + 1;
          }
        }
        // §17.5 #7 — fold any per-flow solver override the child
        // declares into the inherited config. Step / tolerance
        // fields override; algorithm / type stay global.
        SolverConfig ChildSolver = Solver;
        if (Child->Solver) {
          if (Child->Solver->MaxStep != "auto")
            ChildSolver.MaxStep = Child->Solver->MaxStep;
          if (Child->Solver->MinStep != "auto")
            ChildSolver.MinStep = Child->Solver->MinStep;
          ChildSolver.RelTol = Child->Solver->RelTol;
          ChildSolver.AbsTol = Child->Solver->AbsTol;
        }
        Stack.push_back(*FlowId);
        auto ChildG = expand(*Child, Prefix + N.Id + "/", Stack,
                             ChildEnable, ChildEdge, ChildMask,
                             ChildSolver);
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
      // §17.5 #7 — capture the active per-flow solver's step +
      // tolerance overrides on the leaf so the runtime can pick
      // min over all blocks at construction.
      double MaxStepOv = std::numeric_limits<double>::quiet_NaN();
      if (Solver.MaxStep != "auto") {
        try { MaxStepOv = std::stod(Solver.MaxStep); }
        catch (...) {}
      }
      double RelTolOv = Solver.RelTol;
      double AbsTolOv = Solver.AbsTol;
      G.Nodes.push_back({Prefix + N.Id, &N, LeafEnable, LeafEdge,
                         MaskBindings, MaxStepOv, RelTolOv, AbsTolOv});
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
    // Item-3 — substitute `${name}` placeholders in this block's
    // params with the active mask bindings inherited from the
    // enclosing subsystem. Unbound placeholders are left untouched
    // (matches Simulink: a missing mask param surfaces as a
    // visible string in the param, which downstream parsers like
    // polyDegree will then misread — that's the user's signal that
    // a mask binding is missing).
    if (!FN.MaskBindings.empty()) {
      for (auto &P : B.Params) {
        std::string &V = P.second;
        size_t I = 0;
        while (I < V.size()) {
          size_t Open = V.find("${", I);
          if (Open == std::string::npos) break;
          size_t Close = V.find('}', Open + 2);
          if (Close == std::string::npos) break;
          std::string Key = V.substr(Open + 2, Close - Open - 2);
          auto It = FN.MaskBindings.find(Key);
          if (It != FN.MaskBindings.end()) {
            V.replace(Open, Close - Open + 1, It->second);
            I = Open + It->second.size();
          } else {
            I = Close + 1;
          }
        }
      }
    }
    B.Loc = N.Loc;
    B.SampleClass = KI->Sample;
    B.EnableSource = FN.EnableStamp;
    B.EnableEdgeTriggered = FN.EdgeTriggered;
    B.MaxStepOverride = FN.MaxStepOverride;
    B.RelTolOverride  = FN.RelTolOverride;
    B.AbsTolOverride  = FN.AbsTolOverride;
    if (auto *L = N.getData("log_signal"))
      B.LogSignal = (*L == "true");

    // Item-1 — initial output width per kind. Defaults to 1
    // (scalar) and gets refined later by the width-inference pass
    // for blocks whose output width is a function of their input.
    B.OutWidth = 1;
    if (N.Kind == "signal_constant") {
      // `value` may be a vector literal like "[1 2 3]", a matrix
      // literal like "[1 2; 3 4]", or a scalar. Count elements
      // per row (split by `,`/space) and rows (split by `;`).
      // Empty or unparseable falls through to width 1.
      if (auto *V = N.getParam("value")) {
        std::string S = *V;
        auto A = S.find('[');
        auto B0 = S.rfind(']');
        if (A != std::string::npos && B0 != std::string::npos && A < B0) {
          std::string Inner = S.substr(A + 1, B0 - A - 1);
          // Walk row-by-row counting tokens; rows are `;`-separated.
          int Rows = 0;
          int Cols = 0;
          int CurCols = 0;
          bool InTok = false;
          auto endRow = [&]() {
            if (CurCols == 0) return;
            ++Rows;
            if (CurCols > Cols) Cols = CurCols;
            CurCols = 0;
          };
          for (size_t I = 0; I <= Inner.size(); ++I) {
            char C = (I < Inner.size()) ? Inner[I] : ';';
            if (C == ';') {
              if (InTok) { ++CurCols; InTok = false; }
              endRow();
            } else if (C == ',' || C == ' ' || C == '\t') {
              if (InTok) { ++CurCols; InTok = false; }
            } else {
              InTok = true;
            }
          }
          if (Rows > 0 && Cols > 0) {
            B.OutRows  = Rows;
            B.OutCols  = Cols;
            B.OutWidth = Rows * Cols;
          }
        }
      }
      // §17.5 #9 — explicit `shape` override (e.g. for blocks that
      // get their value from elsewhere; not applicable here but
      // surfaced via `params.shape = "rows,cols"` if present).
      if (auto *Sh = N.getParam("shape")) {
        int R = 0, C = 0;
        if (std::sscanf(Sh->c_str(), "%d,%d", &R, &C) == 2 &&
            R > 0 && C > 0) {
          B.OutRows = R; B.OutCols = C;
          if (R * C > B.OutWidth) B.OutWidth = R * C;
        }
      }
    } else if (N.Kind == "signal_psk_mod" || N.Kind == "signal_qam_mod") {
      // Communications (#343) — a modulator maps a symbol index to a complex
      // constellation point, carried as a width-2 [I, Q] vector signal.
      B.OutWidth = 2;
      B.OutRows = 1;
      B.OutCols = 2;
    } else if (N.Kind == "signal_lqr") {
      // Control (#343) — output u = -K·x has one element per row of the gain
      // matrix K (scalar for a 1×N state-feedback row).
      int Kr = matrixRows(N.getParam("K") ? *N.getParam("K") : "");
      B.OutWidth = Kr > 0 ? Kr : 1;
      B.OutRows = B.OutWidth;
      B.OutCols = 1;
    } else if (N.Kind == "signal_dnn_predict") {
      // Deep Learning (#343) — output dim = rows of the output-layer weight W2.
      int Mr = matrixRows(N.getParam("W2") ? *N.getParam("W2") : "");
      B.OutWidth = Mr > 0 ? Mr : 1;
      B.OutRows = B.OutWidth;
      B.OutCols = 1;
    } else if (N.Kind == "signal_image_source") {
      // Vision (#343) — a 2-D grayscale image source. Shape comes from `rows`
      // and `cols`; the output is the flattened row-major width rows·cols.
      // Conformance: if `data` is given, its element count must equal rows·cols.
      int R = N.getParam("rows") ? std::atoi(N.getParam("rows")->c_str()) : 0;
      int C = N.getParam("cols") ? std::atoi(N.getParam("cols")->c_str()) : 0;
      if (R > 0 && C > 0) {
        B.OutRows = R;
        B.OutCols = C;
        B.OutWidth = R * C;
        if (auto *D = N.getParam("data")) {
          // Count numeric tokens (any run of non-separator chars is one pixel).
          int n = 0;
          bool inTok = false;
          for (char ch : *D) {
            bool sep = (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' ||
                        ch == '[' || ch == ']');
            if (!sep && !inTok) ++n;
            inTok = !sep;
          }
          if (n > 0 && n != R * C) {
            Diag.error(B.Loc, "signal_image_source \"" + B.Id +
                                  "\": data has " + std::to_string(n) +
                                  " pixels but shape is " + std::to_string(R) +
                                  "×" + std::to_string(C) + " (" +
                                  std::to_string(R * C) + ")");
            return std::nullopt;
          }
        }
      } else {
        B.OutWidth = 1; // unshaped → scalar fallback
      }
    } else if (N.Kind == "signal_rl_agent") {
      // RL (#343) — a discrete policy emits a scalar action index (argmax); a
      // continuous policy emits one bounded action per output (rows of W2).
      const std::string *AT = N.getParam("actionType");
      bool Discrete = !AT || *AT == "discrete";
      int Mr = matrixRows(N.getParam("W2") ? *N.getParam("W2") : "");
      B.OutWidth = Discrete ? 1 : (Mr > 0 ? Mr : 1);
      B.OutRows = B.OutWidth;
      B.OutCols = 1;
    } else if (N.Kind == "signal_psk_demod" || N.Kind == "signal_qam_demod") {
      // A demodulator's output is always a scalar symbol index, regardless of
      // its width-2 I/Q input — stamp it explicitly so the width-inference
      // pass doesn't make it inherit the input's width.
      B.OutWidth = 1;
      B.OutRows = 1;
      B.OutCols = 1;
    } else if (N.Kind == "signal_window" || N.Kind == "signal_spectrum") {
      // DSP (#343) — windowing / power spectrum. Output width = frame size `n`;
      // if `n` is absent the width inherits the input element count (sentinel 0).
      int Nf = N.getParam("n") ? std::atoi(N.getParam("n")->c_str()) : 0;
      B.OutWidth = Nf > 0 ? Nf : 0;
    } else if (N.Kind == "signal_fft") {
      // DSP (#343) — frame DFT. A real N-point frame maps to a complex
      // spectrum carried as [Re_0..Re_{N-1}, Im_0..Im_{N-1}], width 2N.
      // `n` (frame size) is required to stamp a deterministic width.
      int Nf = N.getParam("n") ? std::atoi(N.getParam("n")->c_str()) : 0;
      B.OutWidth = Nf > 0 ? 2 * Nf : 0;
    } else if (N.Kind == "signal_ifft") {
      // DSP (#343) — frame IDFT. A complex [Re;Im] width-2N frame maps back
      // to a real N-point frame.
      int Nf = N.getParam("n") ? std::atoi(N.getParam("n")->c_str()) : 0;
      B.OutWidth = Nf > 0 ? Nf : 0;
    } else if (N.Kind == "signal_reshape") {
      // §17.5 #9 — reshape (rows, cols). Total element count must
      // match the input; width inference picks the input's element
      // count up at the resolution pass below, the per-kind
      // dispatch here just stamps the shape so downstream blocks
      // see (rows × cols).
      int R = 0, C = 0;
      if (auto *Sh = N.getParam("shape")) {
        std::sscanf(Sh->c_str(), "%d,%d", &R, &C);
      } else {
        if (auto *RP = N.getParam("rows")) R = std::atoi(RP->c_str());
        if (auto *CP = N.getParam("cols")) C = std::atoi(CP->c_str());
      }
      if (R > 0 && C > 0) {
        B.OutRows = R;
        B.OutCols = C;
        B.OutWidth = R * C;
      } else {
        B.OutWidth = 0; // inherit element count; shape stays (1,W)
      }
    } else if (N.Kind == "signal_mux") {
      // `numInputs` × upstream widths — finalised by width inference
      // once we know what's wired to each input.
      B.OutWidth = 0; // sentinel: "compute from inputs"
    } else if (N.Kind == "signal_bus_creator") {
      // §17.5 #1 — pack N scalar inputs into a named-field vector.
      // `params.field_names` is a comma-separated list; the number
      // of names IS the output width.
      if (auto *FN = N.getParam("field_names")) {
        std::string S = *FN;
        std::string Tok;
        for (size_t I = 0; I <= S.size(); ++I) {
          char C = I < S.size() ? S[I] : ',';
          if (C == ',' || C == ' ' || C == '\t') {
            if (!Tok.empty()) {
              B.FieldNames.push_back(std::move(Tok));
              Tok.clear();
            }
          } else {
            Tok.push_back(C);
          }
        }
      }
      B.OutWidth = B.FieldNames.empty() ? 1 : (int)B.FieldNames.size();
    } else if (N.Kind == "signal_bus_selector") {
      // Always emits one scalar — the element matching its `field`.
      B.OutWidth = 1;
    } else if (N.Kind == "signal_demux") {
      // Width is decided per-output-port. For the Tier-I MVP we
      // restrict signal_demux to a single output (the first
      // element of the input vector) so the dispatch can stay
      // scalar; full N-output demux needs per-port output offsets
      // which is a follow-up.
      B.OutWidth = 1;
    } else {
      // For most blocks, the output width inherits from the input
      // width when this block has a data input. The width-inference
      // pass below propagates from upstream once all blocks are
      // constructed. A sentinel of 0 means "decide later".
      bool ScalarSource =
          N.Kind == "signal_sine" || N.Kind == "signal_step" ||
          N.Kind == "signal_pulse" || N.Kind == "signal_ramp" ||
          N.Kind == "signal_clock" || N.Kind == "signal_chirp" ||
          N.Kind == "signal_noise" ||
          N.Kind == "signal_function_call_generator";
      if (!ScalarSource) B.OutWidth = 0; // inherit
    }

    // State counts + loop-breaker classification.
    B.IsLoopBreaker = KI->LoopBreakerAlways;
    if (N.Kind == "signal_integrator") {
      B.ContStateCount = 1;
    } else if (N.Kind == "signal_pid") {
      // Parallel PID with a first-order derivative filter: two continuous
      // states (integral accumulator + derivative-filter state). Output
      // depends on the current error (Kp and filtered-D both feed through),
      // so PID is not a loop breaker — KI->LoopBreakerAlways is already
      // false; an enclosing loop is broken by the plant's dynamics.
      B.ContStateCount = 2;
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
    } else if (N.Kind == "signal_kalman") {
      // The estimate output is an N-vector (N = rows of A). Kalman state lives
      // in a dedicated per-block stash in MflowLinkSim (not Y_/Z_), so no
      // Cont/Disc state slots — only the output width is stamped here.
      int Ns = matrixRows(N.getParam("A") ? *N.getParam("A") : "");
      B.OutWidth = Ns > 0 ? Ns : 1;
      B.OutRows = B.OutWidth;
      B.OutCols = 1;
    } else if (N.Kind == "signal_relay") {
      // One discrete bit (on / off). The relay is a fixed-in-minor
      // block — its output is read every step, but state transitions
      // only at major-step boundaries (the evaluator skips updates
      // when `Deriv != nullptr`, i.e. inside RK4 substeps).
      B.DiscStateCount = 1;
    } else if (N.Kind == "signal_unit_delay" || N.Kind == "signal_zoh" ||
               N.Kind == "signal_discrete_integrator" ||
               N.Kind == "signal_discrete_filter" ||
               N.Kind == "signal_biquad" ||
               N.Kind == "signal_lowpass" || N.Kind == "signal_highpass" ||
               N.Kind == "signal_dcblock" ||
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
      } else if (N.Kind == "signal_biquad") {
        // A second-order section always carries 2 y-history state slots.
        B.DiscStateCount = 2;
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

  //===--------------------------------------------------------------------===//
  // Item-1 — width inference (per-port).
  //
  // Source blocks already declared their output width above. Every
  // other block has `OutWidth = 0` (inherit). Walk in a fixpoint
  // loop: for each unknown-width block, if at least one of its
  // *known* input widths is consistent, adopt that width. Mux
  // sums its inputs. Width mismatches between inputs at the same
  // block (other than scalar-broadcast) are a sourced diagnostic.
  //===--------------------------------------------------------------------===//
  {
    std::unordered_map<std::string, size_t> IdxOf;
    for (size_t I = 0; I < M.Blocks.size(); ++I)
      IdxOf[M.Blocks[I].Id] = I;
    auto inputWidths = [&](size_t I) {
      std::vector<int> Ws;
      for (auto &E : M.Edges) {
        if (E.ToBlock != M.Blocks[I].Id) continue;
        auto It = IdxOf.find(E.FromBlock);
        if (It == IdxOf.end()) continue;
        Ws.push_back(M.Blocks[It->second].OutWidth);
      }
      return Ws;
    };
    // §17.5 #9 — collect input shapes for the chosen propagation
    // rule. Returns one (rows, cols) pair per upstream edge; the
    // caller picks the first 2-D shape (if any) and broadcasts
    // the rest. Scalar (1,1) inputs broadcast trivially; size
    // mismatches are caught alongside the element-count check.
    auto inputShapes = [&](size_t I)
        -> std::vector<std::pair<int, int>> {
      std::vector<std::pair<int, int>> Sh;
      for (auto &E : M.Edges) {
        if (E.ToBlock != M.Blocks[I].Id) continue;
        auto It = IdxOf.find(E.FromBlock);
        if (It == IdxOf.end()) continue;
        const auto &P = M.Blocks[It->second];
        Sh.emplace_back(P.OutRows > 0 ? P.OutRows : 1,
                        P.OutCols > 0 ? P.OutCols : 1);
      }
      return Sh;
    };
    bool Changed = true;
    int Guard = 0;
    while (Changed && Guard++ < 64) {
      Changed = false;
      for (size_t I = 0; I < M.Blocks.size(); ++I) {
        auto &B = M.Blocks[I];
        if (B.OutWidth != 0) continue;
        auto Ws = inputWidths(I);
        if (Ws.empty()) {
          // Sinkless block with no input — fall back to scalar.
          B.OutWidth = 1;
          Changed = true;
          continue;
        }
        // Skip if any input is still unknown — re-try next pass.
        bool AllKnown = true;
        for (int W : Ws) if (W <= 0) { AllKnown = false; break; }
        if (!AllKnown) continue;
        if (B.Kind == "signal_mux") {
          int Sum = 0;
          for (int W : Ws) Sum += W;
          B.OutWidth = Sum > 0 ? Sum : 1;
        } else {
          // Element-wise broadcast: max(1, max(Ws)). Mixed sizes
          // beyond scalar-vs-vector are an error.
          int W = 1;
          for (int Wi : Ws) if (Wi > W) W = Wi;
          for (int Wi : Ws) {
            if (Wi != 1 && Wi != W) {
              Diag.error(B.Loc, "block \"" + B.Id +
                                    "\": width mismatch on inputs "
                                    "(scalar broadcasting allowed, "
                                    "but found widths " +
                                    std::to_string(Wi) + " and " +
                                    std::to_string(W) + ")");
              return std::nullopt;
            }
          }
          B.OutWidth = W;
        }
        Changed = true;
      }
    }
    // Anything still unknown at this point is in a strongly-
    // connected component with no scalar anchor — default to
    // scalar.
    for (auto &B : M.Blocks)
      if (B.OutWidth == 0) B.OutWidth = 1;

    //===------------------------------------------------------------===//
    // §17.5 #9 — shape inference (rows × cols).
    //
    // Fixpoint loop (mirrors the width-inference walk above):
    // each iteration visits every block whose shape isn't yet
    // settled and propagates from its inputs.  For each upstream
    // edge whose OutCols × OutRows == OutWidth and matches the
    // dominant non-scalar shape, the block inherits (R, C);
    // scalar (1, 1) inputs broadcast freely.  Mismatched non-
    // scalar shapes are a sourced diagnostic except on mux/demux
    // (1-D semantics today).  Reshape blocks were stamped during
    // construction — we only validate the element-count contract
    // here. Blocks that never get a 2-D anchor default to
    // (1 × OutWidth), matching the Item-1 single-row layout.
    //===------------------------------------------------------------===//
    {
      std::vector<bool> Done(M.Blocks.size(), false);
      for (size_t I = 0; I < M.Blocks.size(); ++I) {
        if (M.Blocks[I].OutCols > 1) Done[I] = true;
      }
      bool ShapeChanged = true;
      int ShapeGuard = 0;
      while (ShapeChanged && ShapeGuard++ < 64) {
        ShapeChanged = false;
        for (size_t I = 0; I < M.Blocks.size(); ++I) {
          if (Done[I]) continue;
          auto &B = M.Blocks[I];
          if (B.Kind == "signal_reshape") {
            auto Ws = inputWidths(I);
            int Total = (Ws.empty() ? B.OutWidth : Ws.front());
            if (Total <= 0) continue; // wait for upstream
            if (B.OutRows > 0 && B.OutCols > 0 &&
                B.OutRows * B.OutCols != Total) {
              Diag.error(B.Loc, "signal_reshape \"" + B.Id +
                                    "\": shape " +
                                    std::to_string(B.OutRows) + "×" +
                                    std::to_string(B.OutCols) +
                                    " does not match input element count " +
                                    std::to_string(Total));
              return std::nullopt;
            }
            if (B.OutRows <= 0 || B.OutCols <= 0) {
              B.OutRows = 1;
              B.OutCols = B.OutWidth > 0 ? B.OutWidth : Total;
            }
            Done[I] = true; ShapeChanged = true;
            continue;
          }
          auto Sh = inputShapes(I);
          // Wait for every input to have its shape resolved.
          bool AllKnown = true;
          for (size_t K = 0; K < Sh.size(); ++K) {
            // An input is "resolved" once the source block has been
            // marked Done OR was scalar-by-construction (OutWidth=1).
            // Identify the source for this edge:
            // (regenerate to match inputShapes' index order)
            // Simpler: treat (R,C) = (?, ?) where R*C != OutWidth as
            // still pending.
          }
          // Determine dominant non-scalar shape across the inputs.
          int Rd = 1, Cd = 1;
          bool Mismatch = false;
          for (auto [R, C] : Sh) {
            int Total = R * C;
            if (Total <= 1) continue;
            if (Rd * Cd <= 1) { Rd = R; Cd = C; continue; }
            if (R != Rd || C != Cd) {
              if (B.Kind == "signal_mux" || B.Kind == "signal_demux")
                continue;
              Mismatch = true;
              break;
            }
          }
          if (Mismatch) {
            Diag.error(B.Loc, "block \"" + B.Id + "\": shape mismatch on "
                                  "inputs — only scalar broadcast is "
                                  "supported in §17.5 #9 MVP");
            return std::nullopt;
          }
          if (Rd * Cd > 1 && Rd * Cd == B.OutWidth) {
            B.OutRows = Rd;
            B.OutCols = Cd;
            Done[I] = true; ShapeChanged = true;
          } else {
            // No 2-D anchor found yet — leave Done false so a
            // later iteration can pick up newly-shaped upstream
            // blocks. If a full sweep makes no progress we'll
            // fall out of the while loop with this block stamped
            // 1 × OutWidth below.
          }
          (void)AllKnown;
        }
      }
      // Anything still unresolved falls back to 1-D (1 × OutWidth).
      for (auto &B : M.Blocks) {
        if (B.OutRows <= 0 || B.OutCols <= 0 ||
            B.OutRows * B.OutCols != B.OutWidth) {
          B.OutRows = 1;
          B.OutCols = B.OutWidth > 0 ? B.OutWidth : 1;
        }
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Item-1 — sample-time inheritance.
  //
  // A block with no explicit `data.sample_time` (or with
  // `sample_time: "inherited"`) picks up its sample class from its
  // upstream inputs. Rules:
  //   - any continuous upstream → continuous;
  //   - else any discrete upstream → discrete (period = fastest);
  //   - else fixed-in-minor.
  // Walk in topo-ish order via repeated passes until stable.
  //===--------------------------------------------------------------------===//
  {
    auto isInherit = [](const Node &N) -> bool {
      auto *ST = N.getData("sample_time");
      return ST && *ST == "inherited";
    };
    std::unordered_map<std::string, size_t> IdxOf;
    for (size_t I = 0; I < M.Blocks.size(); ++I)
      IdxOf[M.Blocks[I].Id] = I;
    // Cache which flat-node sources requested inheritance — keyed
    // on the block id, since the lowering has already discarded
    // the flat-node wrapper at this point.
    std::unordered_set<std::string> Inherits;
    for (auto &FN : Flat->Nodes) {
      if (isInherit(*FN.Src)) Inherits.insert(FN.Id);
    }
    if (!Inherits.empty()) {
      bool Changed = true;
      int Guard = 0;
      while (Changed && Guard++ < 64) {
        Changed = false;
        for (size_t I = 0; I < M.Blocks.size(); ++I) {
          auto &B = M.Blocks[I];
          if (!Inherits.count(B.Id)) continue;
          // Already settled this round? Only re-touch if a fresh
          // upstream changed sample class.
          SampleTimeClass Best = SampleTimeClass::FixedInMinor;
          double FastestPeriod = 0.0;
          bool AnyDisc = false;
          for (auto &E : M.Edges) {
            if (E.ToBlock != B.Id) continue;
            auto It = IdxOf.find(E.FromBlock);
            if (It == IdxOf.end()) continue;
            const auto &Up = M.Blocks[It->second];
            if (Up.SampleClass == SampleTimeClass::Continuous) {
              Best = SampleTimeClass::Continuous;
            } else if (Up.SampleClass == SampleTimeClass::Discrete) {
              if (!AnyDisc || Up.SamplePeriod < FastestPeriod) {
                FastestPeriod = Up.SamplePeriod;
                AnyDisc = true;
              }
            }
          }
          if (Best == SampleTimeClass::FixedInMinor && AnyDisc) {
            Best = SampleTimeClass::Discrete;
            if (B.SamplePeriod != FastestPeriod) {
              B.SamplePeriod = FastestPeriod;
              Changed = true;
            }
          }
          if (B.SampleClass != Best) {
            B.SampleClass = Best;
            Changed = true;
          }
        }
      }
    }
  }

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

  // Tier-H / Item-4 — validate every `signal_matlab_fcn`'s body.
  // Two acceptable shapes:
  //   - `params.function_body` (Item 4): full `function y = f(...)`
  //     — parsed by the matlab_llvm lexer / parser, walked by a
  //     small AST interpreter at runtime.
  //   - `params.expression`    (Tier H): a single expression in
  //     u1..uN, t, pi, e, with math / trig builtins.
  // `function_body` wins when both are present (the IDE may emit
  // both to let users mix-and-match).
  for (auto &B : M.Blocks) {
    if (B.Kind != "signal_matlab_fcn") continue;
    auto FB = B.Params.find("function_body");
    if (FB != B.Params.end() && !FB->second.empty()) {
      std::string Err = validateMatlabFunctionBody(FB->second);
      if (!Err.empty()) {
        Diag.error(B.Loc, "signal_matlab_fcn \"" + B.Id +
                              "\": " + Err);
        return std::nullopt;
      }
      continue;
    }
    auto It = B.Params.find("expression");
    if (It == B.Params.end() || It->second.empty()) {
      Diag.error(B.Loc, "signal_matlab_fcn \"" + B.Id +
                            "\" missing data.params.expression "
                            "or data.params.function_body");
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
    std::unordered_set<size_t> AllStuck;
    for (size_t I = 0; I < N; ++I)
      if (InDegree[I] > 0) { U.insert(I); AllStuck.insert(I); }
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

    // Item-2 — `settings.solver.algebraicLoopMethod`:
    //   - "off"            ⇒ hard error (the legacy behaviour);
    //   - "newton" / "trust_region" ⇒ accept the cycle and let the
    //     runtime fixed-point-iterate each step.
    if (M.Solver.AlgebraicLoopMethod == "off") {
      std::string List;
      for (size_t K = 0; K < OnLoop.size(); ++K) {
        if (K) List += ", ";
        List += "\"" + M.Blocks[OnLoop[K]].Id + "\"";
      }
      Diag.error(M.Blocks[OnLoop.front()].Loc,
                 "algebraic loop in signal-flow model: blocks " + List +
                     " form a direct-feedthrough cycle with no loop-breaker "
                     "(set settings.solver.algebraicLoopMethod to "
                     "\"trust_region\" or \"newton\" to solve at runtime)");
      return std::nullopt;
    }

    // Record the loop in the IR. Members are the blocks on the
    // cycle in their topological-stable order. Append every
    // still-stuck block (loop members + their downstream
    // dependants) to ExecOrder so the runtime visits them.
    MflAlgebraicLoop AL;
    AL.Members = OnLoop;
    M.AlgebraicLoops.push_back(std::move(AL));

    std::vector<size_t> StuckOrdered(AllStuck.begin(), AllStuck.end());
    std::sort(StuckOrdered.begin(), StuckOrdered.end());
    for (size_t I : StuckOrdered) M.ExecOrder.push_back(I);
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
    if (B.OutWidth != 1) {
      // §17.5 #9 — surface 2-D shape when present so the
      // `--dry-run` IR dump reflects matrix wires; fall back to
      // the legacy `width=N` form for single-row vectors so the
      // existing dump goldens stay byte-identical.
      if (B.OutCols > 1 && B.OutRows > 1)
        OS << " shape=" << B.OutRows << "x" << B.OutCols;
      else
        OS << " width=" << B.OutWidth;
    }
    if (!B.FieldNames.empty()) {
      OS << " fields={";
      for (size_t K = 0; K < B.FieldNames.size(); ++K) {
        if (K) OS << ",";
        OS << B.FieldNames[K];
      }
      OS << "}";
    }
    // §17.5 #7 — surface per-flow solver overrides on the IR
    // dump so `-simulate --dry-run` confirms the tightening.
    if (!std::isnan(B.MaxStepOverride)) {
      double GlobalH = -1.0;
      try {
        if (M.Solver.MaxStep != "auto")
          GlobalH = std::stod(M.Solver.MaxStep);
      } catch (...) {}
      if (GlobalH < 0 ||
          std::fabs(B.MaxStepOverride - GlobalH) > 1e-15) {
        char Buf[64];
        std::snprintf(Buf, sizeof Buf, " maxStep=%g", B.MaxStepOverride);
        OS << Buf;
      }
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
  // Item-2 — algebraic loops accepted by the lowering.
  if (!M.AlgebraicLoops.empty()) {
    OS << "  algebraic-loops:\n";
    for (auto &L : M.AlgebraicLoops) {
      OS << "    {";
      for (size_t K = 0; K < L.Members.size(); ++K) {
        if (K) OS << ", ";
        OS << M.Blocks[L.Members[K]].Id;
      }
      OS << "}\n";
    }
  }
  OS << "  edges:\n";
  for (auto &E : M.Edges)
    OS << "    " << E.Id << " " << E.FromBlock << ":" << E.FromPort << " -> "
       << E.ToBlock << ":" << E.ToPort << "\n";
}

std::vector<std::pair<std::string, bool>> listSignalKinds() {
  std::vector<std::pair<std::string, bool>> Out;
  const auto &Table = kindTable();
  Out.reserve(Table.size());
  for (const auto &[Name, Info] : Table)
    Out.emplace_back(Name, Info.Supported);
  return Out; // std::map iterates in sorted key order.
}

} // namespace matlab::flowchart
