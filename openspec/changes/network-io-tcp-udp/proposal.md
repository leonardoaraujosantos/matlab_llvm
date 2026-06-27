## Why

Today every simulation this compiler runs is a closed box: an mflowlink diagram or
interpreted MATLAB program computes entirely in-process and the only way out is a
file (CSV/HTML) written after the run finishes. There is no socket code anywhere in
the runtime. Users want simulations to **talk to each other while running** — a
controller in one process driving a plant in another, a hardware-in-the-loop rig
exchanging samples, or an mflowlink **3-D environment streaming its live actor
transforms to an external client to capture/visualise**. MATLAB's own answer to this
is `tcpclient`/`tcpserver`/`udpport` (base MATLAB + Instrument Control Toolbox); we
should offer the same surface, on both the MATLAB and the block-diagram sides.

## What Changes

- **MATLAB networking objects** — new handle classes backed by C++ runtime sockets,
  following the same classdef-prelude + parser-fold + Resolver + LowerTensorOps +
  runtime recipe used by `sim3d.World`/`Actor`:
  - `tcpclient(address, port)` — connect to a TCP server; `write(t, data)`,
    `read(t, count)`, `writeline(t, str)`, `readline(t)`, `flush(t)`.
  - `tcpserver(address, port)` — accept one TCP client; same read/write surface.
  - `udpport(...)` — connectionless datagrams; `write(u, data, dst, port)`,
    `read(u, count)`, `writeline`/`readline`, `flush`.
  - Tier 1 is **synchronous, single-connection, non-blocking with a bounded
    timeout**; name-value constructor options and async (`configureCallback`)
    reads are explicitly deferred.
- **mflowlink networking blocks** — new `signal_*` block kinds registered in the
  block catalogue and evaluated by the simulation engine, exchanging signals over
  a socket **only at major (output) step boundaries** so the deterministic RK4
  integration is untouched:
  - `signal_udp_send` / `signal_udp_recv` — stream a scalar/vector wire to/from a
    UDP endpoint.
  - `signal_tcp_send` / `signal_tcp_recv` — same over a TCP stream.
  - Receive blocks are loop-breakers (their output does not depend on this step's
    input), and hold-last-value on starvation so a stalled peer cannot deadlock the
    solver.
- **3-D streaming use case** — because `signal_actor3d` already publishes its 9-wide
  `[tx ty tz rx ry rz sx sy sz]` transform on a wire, wiring it into a
  `signal_udp_send` lets an mflowlink 3-D scene broadcast its live keyframes to any
  external client without new viewer machinery.
- **Examples + docs** — a paired "two sims over a socket" example (a sender and a
  receiver `.mflow`/`.m`), a 3-D-environment-streaming example, and an entry in the
  block/feature docs. A **deterministic loopback** test strategy (both ends bound on
  `127.0.0.1`, fixed payloads, drained at major steps) so the suites stay
  reproducible despite real sockets.

No existing behavior changes; this is purely additive. **Not** in scope for this
change: TLS/encryption, async callbacks, multi-client `tcpserver` fan-out,
serial/`serialport`, transpiling socket blocks to the `-emit-{c,cpp,sv}` lanes, and
binary-protocol framing helpers (left to upstream `signal_math_fcn`/`matlab_fcn`).

## Capabilities

### New Capabilities
- `network-io-matlab`: MATLAB-level TCP/UDP handle objects (`tcpclient`,
  `tcpserver`, `udpport`) with read/write/line/flush methods, runtime socket state
  keyed by the handle, string + numeric-array payloads, and bounded-timeout
  non-blocking semantics.
- `network-io-mflowlink`: mflowlink block-diagram socket blocks
  (`signal_udp_send`/`signal_udp_recv`, `signal_tcp_send`/`signal_tcp_recv`) — their
  registration in the block catalogue, per-step evaluation contract, major-step-only
  I/O, loop-breaker/hold-last-value semantics, and the deterministic 3-D streaming
  path.

### Modified Capabilities
<!-- None: the .mflow loader/schema is additive and the existing flowchart-frontend
     requirements are unchanged. New block kinds are introduced as a new capability. -->

## Impact

- **New runtime:** `runtime/toolbox/instrument/runtime_instrument.cpp` (sockets keyed
  by `matlab_obj*`), `runtime/toolbox/instrument/instrument_classdefs.m`.
- **Frontend wiring (MATLAB objects):** `lib/Parse/Parser.cpp` (fold),
  `lib/Sema/Resolver.cpp` (builtin registration), `lib/MLIR/Passes/LowerTensorOps.cpp`
  (signature table), `tools/matlabc/main.cpp` (prelude discovery: `kToolboxDirs`,
  `Want[]`, `userMentionsExtClasses`, `extClassLeaf`).
- **mflowlink engine:** `lib/Flowchart/SignalFlowLowering.cpp` (`kindTable`),
  `include/matlab/Flowchart/MflowLinkSim.h` (per-block socket state),
  `lib/Flowchart/MflowLinkSim.cpp` (constructor/`reset`/`evalAll` dispatch + major-step
  flush/drain), `test/Flowchart/BlockKindParity/registered_block_kinds.txt`.
- **Platform:** POSIX sockets (`<sys/socket.h>`, `netinet/in.h`, `poll`); the project
  targets Linux/macOS. No new third-party dependency.
- **Risk:** real I/O is non-deterministic and can block — mitigated by non-blocking
  sockets, bounded timeouts, hold-last-value on starvation, and a loopback-only test
  strategy. Documented as "not thread-safe; one handle per thread."
- **Docs:** `docs/mflowlink_blocks.md`, a new instrument-objects doc/README, and
  `examples/` for both surfaces.
