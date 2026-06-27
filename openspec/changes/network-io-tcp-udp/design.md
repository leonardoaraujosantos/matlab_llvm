## Context

The runtime has **no socket or external-I/O code today** — a grep for `AF_INET`,
`bind`, `connect`, `sendto` across the tree returns nothing. Both execution surfaces
are closed boxes:

- **MATLAB programs** (interpreted REPL and the `-emit-*` compiled lanes) compute
  in-process; the only output channels are stdout and files (`fopen`/`fprintf`, the new
  `writematrix`/`sim3d.capture`).
- **mflowlink diagrams** are stepped by `MflowLinkSim` (`lib/Flowchart/MflowLinkSim.cpp`,
  `include/matlab/Flowchart/MflowLinkSim.h`). `MflowLinkSim::evalAll(T, State, Deriv)` is
  a big dispatch over block kinds; continuous blocks publish derivatives integrated by a
  fixed-step RK4 (ode4). The engine's headline property is **deterministic replay** —
  the same diagram + inputs produce byte-identical CSV every run, and the 3-D co-sim
  physics is deliberately in-process with no network anywhere.

Two established patterns are reused wholesale:

1. **Stateful handle objects** (worked example: `sim3d.World`/`Actor`): classdef prelude
   `.m` → parser fold (`lib/Parse/Parser.cpp`) → Resolver builtin registration
   (`lib/Sema/Resolver.cpp`) → LowerTensorOps signature table
   (`lib/MLIR/Passes/LowerTensorOps.cpp`) → runtime functions keyed by `matlab_obj*` in a
   `std::map` → prelude discovery in `tools/matlabc/main.cpp` (`kToolboxDirs`, `Want[]`,
   `userMentionsExtClasses`, `extClassLeaf`).
2. **New block kind** (worked example: `signal_sensor3d`/`signal_collision3d`): register in
   `SignalFlowLowering.cpp kindTable()` + the parity snapshot, add an `evalAll` branch,
   hold per-block state on the sim object, initialise in the constructor/`reset()`.

The request: let simulations talk to each other (MATLAB↔MATLAB, mflowlink↔mflowlink, or
either↔an external client), and in particular let an mflowlink **3-D environment stream
its live actor transforms** to an external capture client. MATLAB's vocabulary for this
is `tcpclient`/`tcpserver`/`udpport`.

## Goals / Non-Goals

**Goals:**
- A `tcpclient`/`tcpserver`/`udpport` MATLAB object surface with
  `write`/`read`/`writeline`/`readline`/`flush`, behaving identically in interpreted and
  compiled lanes.
- `signal_udp_send`/`signal_udp_recv`/`signal_tcp_send`/`signal_tcp_recv` mflowlink blocks
  that exchange wire values **without breaking deterministic RK4 stepping**.
- A 3-D-environment → UDP/TCP streaming path that reuses the existing
  `signal_actor3d` 9-wide transform wire.
- A loopback-based, reproducible test strategy and paired examples.

**Non-Goals (this change):**
- TLS/encryption, authentication.
- Async/event-driven reads (`configureCallback`), name-value constructor options.
- Multi-client `tcpserver` fan-out (Tier 1 accepts exactly one client).
- `serialport` and other Instrument Control transports.
- Transpiling socket blocks/objects to `-emit-{c,cpp,python,ts,sv}` (runtime-only for
  now; the codegen lanes get skip-markers).
- Binary protocol framing/typing beyond a default byte/double payload (compose with
  `signal_math_fcn` / `matlab_fcn`).
- Thread safety (the interpreter/JIT runs user code single-threaded; documented as
  "one handle per thread").

## Decisions

### D1. POSIX sockets, non-blocking, bounded timeout — not blocking I/O

Use raw POSIX sockets (`socket`/`bind`/`listen`/`accept`/`connect`/`send`/`recv`/`sendto`/
`recvfrom`) set `O_NONBLOCK`, with `poll(2)` and a small bounded timeout (default ~100 ms,
configurable) on reads. **Why:** the codebase has no threading model and all runtime state
is global/non-thread-safe; a blocking `recv` would freeze the interpreter or the solver
indefinitely. Non-blocking + bounded `poll` keeps a stalled or absent peer from
deadlocking a run. *Alternatives rejected:* (a) blocking sockets — simplest but can hang a
simulation forever, unacceptable for a deterministic engine; (b) a background reader thread
+ ring buffer — better ergonomics but introduces the project's first threading and locking,
too large for Tier 1; deferred to a Tier-2 async follow-on. No Boost/ASIO dependency — bare
syscalls keep the build dependency-free.

### D2. Socket I/O only at major-step boundaries (mflowlink)

Send/receive happen in `evalAll` **only on the major (output) step**, never inside an RK4
minor stage. `evalAll` already runs for both; the socket branches gate on the same
"output step" condition used by sinks like `signal_scope`/`logSample`. **Why:** RK4
evaluates the diagram 4× per step at intermediate states; transmitting those intermediate
values (or consuming a fresh packet mid-stage) would inject non-physical, non-deterministic
data into the integrator. Confining I/O to major steps keeps the continuous solution a pure
function of the integrator state. Receive blocks are **direct-feedthrough-free**
(loop-breakers): they latch incoming data at the major step and hold it across the minor
stages, exactly like `signal_unit_delay`/`signal_zoh` hold their state.

### D3. Receive = loop-breaker + hold-last-value on starvation

A `*_recv` block's output does not depend on the current step's input, so it is registered
with `LoopBreakerAlways = true` (same flag as integrator/unit_delay/zoh). On each major
step it drains all currently-available datagrams/bytes (non-blocking) and latches the last
one; if nothing arrived, it re-emits the previously latched value (or the `initialValue`
param before any packet). **Why:** a control loop reading a remote sensor must not throw an
algebraic-loop error, and a slow/absent peer must not stall the fixed-rate solver. *Trade-off:*
the simulation does not wait for the peer — it runs at its own clock and uses the freshest
value seen. True lock-step co-simulation (block until the peer produces step N) is a
Non-Goal here and noted as an Open Question.

### D4. Runtime state keyed by handle (MATLAB) / by block index (mflowlink)

MATLAB objects: `std::map<matlab_obj*, Conn>` in
`runtime/toolbox/instrument/runtime_instrument.cpp`, mirroring `g_worlds`/`g_actors`. Each
`Conn` holds the fd, role, peer address, input buffer, and timeout. The constructor
(`matlab_tcpclient_new(obj, host, port)`) connects eagerly; `matlab_obj*` is the stable key.
mflowlink blocks: a `std::vector<SocketSlot>` on `MflowLinkSim` indexed like the other
per-block state arrays (`TransportBuf_`, `Kalman_`), opened in `reset()`, closed in the
destructor. **Why:** these are the two state-ownership idioms already in the codebase; not
inventing a third.

### D5. Payload marshalling reuses existing types

`write(t, v)` takes a `matlab_mat*` and sends its element bytes (Tier-1 default = the
matrix's doubles, little-endian, documented; an optional `datatype` arg is a Tier-2
nicety). `read(t, n)` returns a `mat_alloc(1, n)` row. `writeline`/`readline` use
`matlab_string*` (the `toStr` helper from `runtime_sim3d.cpp` and `matlab_string_from_literal`
for returns). **Why:** identical to how sim3d marshals vectors/strings and how
`sim3d.capture` returns a fresh matrix — no new marshalling infrastructure. *Note:* sending
raw doubles is not wire-compatible with arbitrary external protocols; that is acceptable for
sim↔sim (both ends use the same runtime) and is the documented Tier-1 contract. Byte-exact
interop with third-party tools is composed via upstream packing blocks.

### D6. Constructor arguments positional-only in Tier 1

`tcpclient("addr", port)`, `tcpserver("addr", port)`, `udpport()` /
`udpport("LocalPort", p)` accept positional string+number args, forwarded straight from the
classdef constructor to the runtime `*_new` builtin (the `sim3d_Actor(name, shape)` pattern).
Name-value options (`'Timeout'`, `'ByteOrder'`, …) are **deferred** — the compiler has no
runtime name-value parsing inside constructors (all existing name-value handling is
compile-time at Lowering, e.g. `optimoptions`). A single `'LocalPort'` positional/keyword for
`udpport` is the one ergonomic exception, parsed at Lowering if needed. **Why:** keeps Tier 1
small and avoids a runtime name-value mechanism that does not exist yet.

### D7. Deterministic loopback test strategy

All regression fixtures bind both endpoints on `127.0.0.1`, use fixed payloads, and have the
receiver drain at major steps. The mflowlink loopback test runs a `*_send` and a `*_recv`
in the **same** diagram (or two diagrams in one harness process) so no external server is
needed and the byte stream is fully determined. MATLAB-object tests pair a `tcpserver` and a
`tcpclient` (or two `udpport`s) within one test program. **Why:** real sockets are
non-deterministic in general, but a self-contained loopback with fixed data and major-step
draining is reproducible; this is what keeps the suites green. Tests that would depend on OS
scheduling/timing are avoided (no asserting *how many* packets arrived under load, only the
*values* once delivered).

## Risks / Trade-offs

- **Non-determinism / flakiness from real I/O** → confine to loopback + fixed payloads +
  major-step draining (D7); assert on delivered values, never on timing; mark socket
  fixtures so they can be skipped in constrained CI sandboxes if loopback is unavailable.
- **A peer that never sends could stall a run** → non-blocking sockets + bounded `poll`
  timeout + hold-last-value (D1, D3); the solver never blocks on the network.
- **RK4 minor-stage I/O would corrupt the integration** → major-step-only gate (D2),
  receive blocks latch-and-hold like unit_delay.
- **Port already in use / connection refused** → constructors fail loudly (spec requirement),
  return a clear runtime error; `reset()` for blocks reports a bind failure rather than
  silently producing zeros.
- **Sandboxed/locked-down CI may forbid `bind`/`connect` even on loopback** → keep the
  socket fixtures behind a capability check / skip-marker so the core suites still pass;
  document the env requirement.
- **Raw-double payloads aren't a real wire protocol** → documented Tier-1 contract; sim↔sim
  is the supported interop; third-party byte-exactness is out of scope and composed upstream.
- **First use of POSIX networking headers in the build** → guarded to Linux/macOS (the
  project's targets); Windows is not a current target. No new third-party dependency.

## Migration Plan

Purely additive; no migration. Rollout is tiered so each tier is independently shippable and
testable:

1. **Tier 1a — MATLAB UDP + TCP client/server**: runtime socket layer + `udpport`,
   `tcpclient`, `tcpserver` objects with `write`/`read`/`writeline`/`readline`/`flush`;
   loopback tests; one "two sims over a socket" MATLAB example. Codegen lanes get
   skip-markers.
2. **Tier 1b — mflowlink socket blocks**: `signal_udp_send/recv`, `signal_tcp_send/recv` in
   `kindTable` + parity snapshot + `evalAll` + sim-side socket slots; loopback send→recv
   fixture; block docs.
3. **Tier 1c — 3-D streaming example**: wire a `signal_actor3d` transform into a
   `signal_udp_send`; a small standalone receiver script that captures the transform stream;
   doc the pattern.

Rollback is removing the additive files/registrations; nothing existing depends on them.

## Implementation notes (discovered during build)

- **Method dispatch is dot-syntax only.** This frontend dispatches classdef
  methods via `obj.method(args)`, not the bare `method(obj, args)` function form
  (the latter resolves `method` as a global name → "undefined name"). The
  networking objects therefore document and use `c.write(data)` / `s.read(n)`.
- **One class per prelude file.** A handle class whose method `count` param is
  unused in a program mis-infers to `void*` in the C++ emitter and collides with
  the runtime extern. Splitting `tcpclient`/`tcpserver`/`udpport` into one file
  each (`instrument_class_*.m`) means a single-class program never drags in the
  others' identically-named methods. (Same reason the loopback test and example
  use `udpport` for both endpoints — a single class.)
- **Compiled-lane text I/O is deferred.** `read(count)` (a matrix) is wrapped as
  a Matrix in the emitter so it disp's correctly; `readline` returns a
  `matlab_string*` through a classdef method, and the emitter's string-disp
  rewrite only fires on a *direct* string-producing call, not a method return.
  Numeric `write`/`read` have full interpreted↔compiled parity; `writeline`/
  `readline` are interpreter-complete, with the compiled string path left to a
  follow-on (it needs a "string-returning method" typing pass in EmitC).

## Open Questions

- **Lock-step co-simulation?** Tier 1 runs each side at its own clock (hold-last-value). Do
  we want an optional *blocking, barrier-synchronised* mode where step N waits for the peer's
  step N? That needs a handshake protocol and reintroduces blocking — likely a separate
  future change.
- **Payload datatype negotiation.** Default is raw doubles. Is a Tier-2 `datatype`/`ByteOrder`
  option (uint8/int16/single, endianness) worth it for external-tool interop, or do we keep
  sim↔sim only?
- **Should `udpport` support multicast/broadcast** for the "one 3-D environment, many capture
  clients" fan-out, or is unicast-per-client sufficient for Tier 1?
- **CI socket policy.** Confirm the CI sandbox permits loopback `bind`/`connect`; if not,
  finalise the skip-marker/capability-gate so the networking suite is opt-in.
