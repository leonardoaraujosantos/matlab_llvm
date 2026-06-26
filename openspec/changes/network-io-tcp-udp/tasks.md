## 1. Runtime socket layer (shared)

- [ ] 1.1 Add `runtime/toolbox/instrument/` to the build (CMake source glob + include path), mirroring `runtime/toolbox/sim3d/`.
- [ ] 1.2 Implement a small internal socket helper in `runtime_instrument.cpp`: non-blocking `socket`/`bind`/`listen`/`accept`/`connect`, `poll`-with-timeout read, `send`/`sendto`, and a `recv`/`recvfrom` drain into an input buffer. POSIX-only, guarded for Linux/macOS, no third-party deps.
- [ ] 1.3 Define the `Conn` state struct (fd, role, peer addr, input buffer, timeout) and the `std::map<matlab_obj*, Conn>` registry + a `connOf(handle)` accessor (mirror `actorOf`).
- [ ] 1.4 Reuse the `toStr` string-decode helper and `mat_alloc`/`matlab_string_from_literal` marshalling for payloads.

## 2. MATLAB objects — `tcpclient` / `tcpserver` / `udpport` (Tier 1a)

- [ ] 2.1 Write `runtime/toolbox/instrument/instrument_classdefs.m`: `< handle` classes forwarding `obj` (+ string/number/matrix args) to `matlab_*` builtins; methods `write`, `read`, `writeline`, `readline`, `flush`.
- [ ] 2.2 Implement the runtime builtins: `matlab_tcpclient_new(obj, host, port)`, `matlab_tcpserver_new(obj, host, port)`, `matlab_udpport_new(obj, localPort)`; `matlab_net_write(obj, mat)`, `matlab_net_read(obj, count)→mat`, `matlab_net_writeline(obj, str)`, `matlab_net_readline(obj)→str`, `matlab_net_flush(obj)`, plus `matlab_udp_write_to(obj, mat, host, port)`. Eager connect; fail loudly on construct error.
- [ ] 2.3 Register every `matlab_*` builtin name in the `lib/Sema/Resolver.cpp` known-builtins set.
- [ ] 2.4 Add the signature-table entries in `lib/MLIR/Passes/LowerTensorOps.cpp` (`{name, sym, retTy, {argTys}}`); register `matlab_net_read`/`matlab_net_readline` returns in EmitC `MatrixReturningFns` only if reached as a free function (object methods are typed via the classdef getter — verify).
- [ ] 2.5 Add the parser fold in `lib/Parse/Parser.cpp` if a package prefix is used; otherwise confirm bare `tcpclient(...)` resolves to the prelude classdef constructor.
- [ ] 2.6 Wire prelude discovery in `tools/matlabc/main.cpp`: add `"instrument"` to both `kToolboxDirs`, add `Want[]` entries, add the names to `userMentionsExtClasses`, and the `extClassLeaf` mapping.
- [ ] 2.7 Add `.skip-emit-{c,cpp,python,typescript}` markers for the networking fixtures (runtime-only surface).

## 3. mflowlink socket blocks (Tier 1b)

- [ ] 3.1 Register `signal_udp_send`, `signal_udp_recv`, `signal_tcp_send`, `signal_tcp_recv` in `lib/Flowchart/SignalFlowLowering.cpp kindTable()` (Known+Supported; `*_recv` with `LoopBreakerAlways=true`).
- [ ] 3.2 Add the four kinds to `test/Flowchart/BlockKindParity/registered_block_kinds.txt`.
- [ ] 3.3 Add a `std::vector<SocketSlot>` (fd + peer + input buffer + last-value latch) to `include/matlab/Flowchart/MflowLinkSim.h`, indexed like other per-block state.
- [ ] 3.4 In the `MflowLinkSim` constructor/block-classification, allocate a slot per socket block and read `host`/`port`/`initialValue` params; open sockets in `reset()`, close in the destructor.
- [ ] 3.5 Add `evalAll` branches: `*_send` reads its input wire and transmits **only on the major/output step**; `*_recv` drains available data non-blocking at the major step, latches the last value, and outputs the latched (or initial) value every step.
- [ ] 3.6 Verify minor RK4 stages perform no socket I/O (gate on the same output-step condition as `logSample`).

## 4. 3-D environment streaming (Tier 1c)

- [ ] 4.1 Confirm a `signal_actor3d` 9-wide transform wire can feed a `signal_udp_send` input port; adjust vector handling if the send block must accept a vector wire.
- [ ] 4.2 Add `examples/mflowlink/3d/` (or alongside existing) a scene that streams an actor's transform over UDP; add a tiny standalone receiver script that prints/captures the frames.

## 5. Examples + docs

- [ ] 5.1 Add a paired "two sims over a socket" example: a sender `.mflow`/`.m` and a receiver `.mflow`/`.m` (loopback) under `examples/`.
- [ ] 5.2 Document the four block kinds (params: `host`, `port`, `initialValue`) in `docs/mflowlink_blocks.md`.
- [ ] 5.3 Add an instrument-objects README/doc for `tcpclient`/`tcpserver`/`udpport` (methods, Tier-1 semantics, "one handle per thread", raw-double payload contract).

## 6. Tests (deterministic loopback)

- [ ] 6.1 MATLAB-object test: a single program pairing `tcpserver`+`tcpclient` (and two `udpport`s) on `127.0.0.1`, asserting round-tripped values; verify interpreted vs compiled parity.
- [ ] 6.2 mflowlink loopback fixture: a diagram (or one-process harness) running `*_send`→`*_recv` with fixed payloads, asserting received values in `test/Flowchart/SimulateRun/run_tests.sh`.
- [ ] 6.3 Gate the socket fixtures behind a capability/skip-marker so the core suites still pass where loopback `bind`/`connect` is disallowed.
- [ ] 6.4 Run the full differential + Run + flowchart suites; confirm no regression and the new fixtures pass.

## 7. Validation

- [ ] 7.1 `openspec validate network-io-tcp-udp --strict` passes.
- [ ] 7.2 Update the OpenSpec baseline index / archive the change per the project flow once merged.
