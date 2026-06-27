## 1. Runtime socket layer (shared)

- [x] 1.1 Add `runtime/toolbox/instrument/` to the build (CMake source glob + include path), mirroring `runtime/toolbox/sim3d/`.
- [x] 1.2 Implement a small internal socket helper in `runtime_instrument.cpp`: non-blocking `socket`/`bind`/`listen`/`accept`/`connect`, `poll`-with-timeout read, `send`/`sendto`, and a `recv`/`recvfrom` drain into an input buffer. POSIX-only, guarded for Linux/macOS, no third-party deps.
- [x] 1.3 Define the `Conn` state struct (fd, role, peer addr, input buffer, timeout) and the `std::map<matlab_obj*, Conn>` registry + a `connOf(handle)` accessor (mirror `actorOf`).
- [x] 1.4 Reuse the `toStr` string-decode helper and `mat_alloc`/`matlab_string_from_literal` marshalling for payloads.

## 2. MATLAB objects — `tcpclient` / `tcpserver` / `udpport` (Tier 1a)

- [x] 2.1 Write the classdef preludes: `< handle` classes forwarding `obj` (+ args) to `matlab_*` builtins; methods `write`/`read`/`writeline`/`readline`/`flush`. (Split into `instrument_class_{tcpclient,tcpserver,udpport}.m` — one per file — so a single-class program does not drag in the others' methods, which mis-type in the C++ emit lane.)
- [x] 2.2 Implement the runtime builtins: `matlab_tcpclient_new`/`matlab_tcpserver_new`/`matlab_udpport_new`, `matlab_net_write`/`matlab_udp_write_to`/`matlab_net_read`/`matlab_net_writeline`/`matlab_net_readline`/`matlab_net_flush`. Eager connect; lazy `accept` for the server.
- [x] 2.3 Register every `matlab_*` builtin name in `lib/Sema/Resolver.cpp`.
- [x] 2.4 Add the signature-table entries in `lib/MLIR/Passes/LowerTensorOps.cpp`; add `matlab_net_read` to EmitC `MatrixReturningFns` so `read(count)` disp's as a row, not a pointer.
- [x] 2.5 Bare class names resolve to the prelude constructor (no parser fold needed); confirmed via `tf`-style registration.
- [x] 2.6 Wire prelude discovery in `tools/matlabc/main.cpp`: `kToolboxDirs` (both copies), `Want[]`, `userMentionsExtClasses`, `extClassLeaf`.
- [~] 2.7 Skip-emit markers — N/A: the only auto-suite fixture is the cpp-only differential `net_udp_loopback.m`; no `test/Run` fixture was added, so no markers needed. (Deferred text-I/O compiled gap noted in design.)

## 3. mflowlink socket blocks (Tier 1b)

- [x] 3.1 Register `signal_udp_send`/`signal_udp_recv`/`signal_tcp_send`/`signal_tcp_recv` in `SignalFlowLowering.cpp kindTable()` (`*_recv` are loop-breakers).
- [x] 3.2 Add the four kinds to the BlockKindParity snapshot (regenerated; 110 kinds).
- [x] 3.3 Add `std::vector<NetSlot>` (fd + listenFd + flags + host/port + last-value latch) to `MflowLinkSim.h`.
- [x] 3.4 Constructor classifies socket blocks + reads params; `openNetSockets()` (recv-before-send two-pass) on `reset()`, `closeNetSockets()` in the destructor.
- [x] 3.5 `evalAll` branches: `*_send` passes its input through; `*_recv` publishes the latch. `commitNetworkIO()` (once per major step) transmits sends + drains recvs.
- [x] 3.6 Socket I/O confined to the major step (commitNetworkIO, not evalAll) — no minor-stage I/O.

## 4. 3-D environment streaming (Tier 1c)

- [x] 4.1 `signal_udp_send` accepts a vector wire (a wide source streams every element).
- [x] 4.2 `examples/mflowlink/3d/net_stream_orbit.mflow` tees an actor's position into `signal_udp_send`; verified with an external Python receiver capturing the `[x y z]` frames.

## 5. Examples + docs

- [x] 5.1 Paired examples: `examples/instrument/udp_loopback.m` (self-contained) + `tcp_sender.m`/`tcp_receiver.m` (two-process; verified across real processes).
- [x] 5.2 Documented the four block kinds in `docs/mflowlink_blocks.md`.
- [x] 5.3 `examples/instrument/README.md` for `tcpclient`/`tcpserver`/`udpport` (methods, Tier-1 semantics, one-handle-per-thread, raw-float64 contract).

## 6. Tests (deterministic loopback)

- [x] 6.1 MATLAB-object parity: `test/Differential/net_udp_loopback.m` (single-class udpport send→read, interpret vs compile). Examples cover tcpclient/tcpserver across processes.
- [x] 6.2 mflowlink loopback: `examples/mflowlink/net_udp_loopback.mflow` + a SimulateRun check (constant → udp_send → 127.0.0.1 → udp_recv → scope delivers 42).
- [x] 6.3 SimulateRun net check gated behind a loopback-bind probe (`python3` UDP bind) so the suite skips where sockets are disallowed.
- [x] 6.4 Differential (15), SimulateRun (300), BlockKindParity (110) all green; full build clean.

## 7. Validation

- [x] 7.1 `openspec validate network-io-tcp-udp --strict` passes.
- [ ] 7.2 Archive the change per the project flow once merged.
