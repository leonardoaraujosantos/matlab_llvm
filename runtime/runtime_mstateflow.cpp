// runtime/runtime_mstateflow.cpp — mStateflow runtime hooks.
//
// The MATLAB lowering (lib/StateChart/Lowering.cpp) emits self-
// contained `<chart>_tick` functions that operate on plain MATLAB
// structs. They don't need a C runtime to execute — `matlabc`
// compiles them through the standard MLIR + LLVM pipeline like any
// `.m` file.
//
// This translation unit reserves a home for the C-side hooks the
// chart runtime will grow into:
//   - Tier 4e — DAP event sinks for stateEnter / stateExit /
//     transitionFired / superStepBegin / superStepEnd /
//     eventBroadcast, drained by the inline DAP server in
//     tools/matlabc/main.cpp.
//   - Tier 6 — snapshot ring helpers (binary-compatible with the
//     mflowLink ring in lib/Flowchart/MflowLinkSim.cpp).
//   - Future — temporal-counter / `in(state)` / chart-fn inlining
//     helpers that need fast in-process state.
//
// All chart-runtime entry points are extern "C" and no-op when no
// listener is attached, so the JIT can resolve them via
// LLJIT's DynamicLibrarySearchGenerator without hard-coding their
// addresses.

#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct ChartEvent {
  enum class Kind {
    StateEnter, StateExit, TransitionFired,
    SuperStepBegin, SuperStepEnd, EventBroadcast,
  };
  Kind K;
  std::string A;          // state id / transition id / event name
  std::string B, C;       // src / dst (transition) or empty
  double T = 0.0;         // sim time
  int Iter = 0;
  bool Quiescent = false;
};

// Process-wide event queue + listener flag. The DAP server in
// `tools/matlabc/main.cpp` flips `g_HasListener` to true on launch
// and drains `g_Events` from its own thread. With no listener
// attached every hook is a near-no-op (one atomic load + early
// return).
struct ChartEventQueue {
  std::mutex M;
  std::vector<ChartEvent> Events;
  bool Listening = false;
};

ChartEventQueue &queue() {
  static ChartEventQueue Q;
  return Q;
}

// Snapshot ring shared *in concept* with mflowLink: per-chart-tick
// the runtime may push a snapshot at super-step boundaries. The
// actual MATLAB-side state struct lives inside the JIT-emitted
// chart_tick frame, so snapshots are tagged blobs of opaque bytes
// here — the front-end serialises the struct on save and restores
// it on read. Capacity matches Stateflow's default 256 (the IDE
// can override via the chart's snapshot config).
struct ChartSnapshotRing {
  std::mutex M;
  size_t Cap = 256;
  std::vector<std::pair<std::string, std::vector<uint8_t>>> Slots;
};

ChartSnapshotRing &snapshotRing() {
  static ChartSnapshotRing R;
  return R;
}

// Bounded FIFO event queue per the runtime contract (§6.4). The
// chart-runtime layer routes broadcasts here so external C callers
// (matlabc-emitted chart_tick, embedded targets, the DAP server's
// pre-step queue) can share a single source of truth. The interpreter
// keeps its own per-step Events_ set so it doesn't pay the lock cost
// each guard evaluation — the FIFO is a transport layer, not the
// authoritative store.
struct ChartEventFIFO {
  std::mutex M;
  std::deque<std::string> Q;
  size_t Cap = 64;
};

ChartEventFIFO &fifo() {
  static ChartEventFIFO F;
  return F;
}

} // namespace

extern "C" {

// Flip on / off whether the runtime should record events. The DAP
// server calls this once on launch so the no-listener fast path is
// the default.
void mstateflow_set_listening(int On) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  Q.Listening = (On != 0);
}

int mstateflow_is_listening(void) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  return Q.Listening ? 1 : 0;
}

void mstateflow_dap_state_enter(const char *Id, double T) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::StateEnter;
  E.A = Id ? Id : ""; E.T = T;
  Q.Events.push_back(std::move(E));
}

void mstateflow_dap_state_exit(const char *Id, double T) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::StateExit;
  E.A = Id ? Id : ""; E.T = T;
  Q.Events.push_back(std::move(E));
}

void mstateflow_dap_transition_fired(const char *Id, const char *Src,
                                     const char *Dst, double T,
                                     const char *Event) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::TransitionFired;
  E.A = Id ? Id : ""; E.B = Src ? Src : ""; E.C = Dst ? Dst : "";
  E.T = T;
  // Stash event name in A's suffix; the drain code splits on '|'.
  if (Event && *Event) { E.A += "|"; E.A += Event; }
  Q.Events.push_back(std::move(E));
}

void mstateflow_dap_superstep_begin(double T, int Iter) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::SuperStepBegin;
  E.T = T; E.Iter = Iter;
  Q.Events.push_back(std::move(E));
}

void mstateflow_dap_superstep_end(double T, int Iter, int Quiescent) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::SuperStepEnd;
  E.T = T; E.Iter = Iter; E.Quiescent = (Quiescent != 0);
  Q.Events.push_back(std::move(E));
}

void mstateflow_dap_event_broadcast(const char *Name, double T) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  if (!Q.Listening) return;
  ChartEvent E; E.K = ChartEvent::Kind::EventBroadcast;
  E.A = Name ? Name : ""; E.T = T;
  Q.Events.push_back(std::move(E));
}

// Drain — the DAP server pops up to MaxOut events, serialises them
// to JSON, and emits the chart-namespaced events. Returns the
// number of events copied; resets the queue when called with
// MaxOut > 0.
//
// The C ABI uses a tag-then-payload encoding so the caller doesn't
// need access to ChartEvent's layout. Each event flushed writes:
//   - 1 byte: kind (0..5 matches ChartEvent::Kind order)
//   - then ASCII text terminated by '\0' for each of A, B, C
//   - then double T (8 bytes, native endian)
//   - then int  Iter (4 bytes)
//   - then int  Quiescent (4 bytes; 0/1)
// We expose a simpler helper: just return a pointer to the queue
// the caller can introspect via the named getters below.
int mstateflow_drain_count(void) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  return static_cast<int>(Q.Events.size());
}

void mstateflow_drain_reset(void) {
  auto &Q = queue();
  std::lock_guard<std::mutex> L(Q.M);
  Q.Events.clear();
}

// Snapshot-ring API. Tier 6 (Snapshots / step-back) wires this
// up. For now: store a named blob under at most `Cap` slots,
// evicting the oldest when full (named entries are pinned — the
// implementation lands with the Tier-6 work).
int mstateflow_snapshot_save_blob(const char *Name, const void *Data,
                                  size_t Len) {
  if (!Name) return 0;
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  std::vector<uint8_t> Buf(Len);
  if (Len) std::memcpy(Buf.data(), Data, Len);
  // Replace any existing slot with the same name.
  for (auto &S : R.Slots) {
    if (S.first == Name) { S.second = std::move(Buf); return 1; }
  }
  if (R.Slots.size() >= R.Cap) R.Slots.erase(R.Slots.begin());
  R.Slots.emplace_back(Name, std::move(Buf));
  return 1;
}

size_t mstateflow_snapshot_size(const char *Name) {
  if (!Name) return 0;
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  for (auto &S : R.Slots) if (S.first == Name) return S.second.size();
  return 0;
}

int mstateflow_snapshot_copy(const char *Name, void *Out, size_t Cap) {
  if (!Name) return 0;
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  for (auto &S : R.Slots) {
    if (S.first != Name) continue;
    size_t N = S.second.size();
    if (N > Cap) return 0;
    std::memcpy(Out, S.second.data(), N);
    return static_cast<int>(N);
  }
  return 0;
}

void mstateflow_snapshot_reset(void) {
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  R.Slots.clear();
}

// --- Event FIFO ------------------------------------------------------
// External C callers (matlabc-emitted chart_tick wrappers, embedded
// firmware drivers) use these to queue events for the next super-
// step. The DAP server keeps using its in-process emit path so its
// stepSuperStep still observes the same events.

void mstateflow_event_set_capacity(int Cap) {
  if (Cap < 1) Cap = 1;
  auto &F = fifo();
  std::lock_guard<std::mutex> L(F.M);
  F.Cap = static_cast<size_t>(Cap);
  while (F.Q.size() > F.Cap) F.Q.pop_front();
}

int mstateflow_event_push(const char *Name) {
  if (!Name) return 0;
  auto &F = fifo();
  std::lock_guard<std::mutex> L(F.M);
  if (F.Q.size() >= F.Cap) F.Q.pop_front();  // bounded; drop oldest
  F.Q.emplace_back(Name);
  return 1;
}

int mstateflow_event_count(void) {
  auto &F = fifo();
  std::lock_guard<std::mutex> L(F.M);
  return static_cast<int>(F.Q.size());
}

// Copies the next pending event name into `Out` (bounded by `Cap`)
// and pops it. Returns 0 when the queue is empty or the buffer is too
// small. The chart-runtime expects callers to drain in a loop before
// invoking the chart's step function.
int mstateflow_event_pop(char *Out, size_t Cap) {
  if (!Out || Cap == 0) return 0;
  auto &F = fifo();
  std::lock_guard<std::mutex> L(F.M);
  if (F.Q.empty()) return 0;
  const std::string &Front = F.Q.front();
  if (Front.size() + 1 > Cap) return 0;
  std::memcpy(Out, Front.data(), Front.size());
  Out[Front.size()] = '\0';
  F.Q.pop_front();
  return 1;
}

void mstateflow_event_reset(void) {
  auto &F = fifo();
  std::lock_guard<std::mutex> L(F.M);
  F.Q.clear();
}

// --- C ABI bridge ----------------------------------------------------
// `mstateflow_tick` per §6.4 of the roadmap. Direct C callers don't
// have access to the matlabc MATLAB-runtime struct types, so the
// canonical C entry point is the matlabc-emitted `<chart>_tick`
// function from `-emit-matlab`/`-emit-c`. This bridge exposes a
// thin shim that orchestrates the event FIFO + announces that a
// super-step is about to run, so external drivers can sync with
// the DAP server's snapshot ring.
//
// Pattern for embedded use:
//   mstateflow_event_push("tick");
//   <user-provided>chart_tick(&state, &inputs, &events);   // emit-c
//   mstateflow_event_reset();
void mstateflow_tick_begin(const char *Chart) {
  static_cast<void>(Chart);  // reserved for per-chart instrumentation.
  // The DAP listener observes `superStepBegin` via the in-process
  // path; an external caller can pair this with `mstateflow_drain_*`
  // to surface the same trace.
}

void mstateflow_tick_end(const char *Chart) {
  static_cast<void>(Chart);
  // Counterpart to _tick_begin — bookmark for instrumentation only.
}

// Introspection — let the DAP layer list every name currently in the
// ring. Returned pointers stay valid until the next save / reset.
int mstateflow_snapshot_count(void) {
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  return static_cast<int>(R.Slots.size());
}

const char *mstateflow_snapshot_name(int Idx) {
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  if (Idx < 0 || static_cast<size_t>(Idx) >= R.Slots.size()) return nullptr;
  return R.Slots[Idx].first.c_str();
}

size_t mstateflow_snapshot_name_size(int Idx) {
  auto &R = snapshotRing();
  std::lock_guard<std::mutex> L(R.M);
  if (Idx < 0 || static_cast<size_t>(Idx) >= R.Slots.size()) return 0;
  return R.Slots[Idx].second.size();
}

} // extern "C"
