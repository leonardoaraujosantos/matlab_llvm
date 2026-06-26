# instrument — TCP/IP + UDP networking objects

These examples drive the `tcpclient` / `tcpserver` / `udpport` command-line
networking objects — a faithful subset of base MATLAB +
[Instrument Control Toolbox](https://www.mathworks.com/help/instrument/) — so
simulations can **talk to each other while running** instead of only writing a
file at the end.

State (the socket) lives in the C++ runtime, keyed by the object handle, so the
objects work in the **interpreted REPL** and the **compiled lanes** alike.

## API

| Call | Meaning |
|------|---------|
| `c = tcpclient(address, port)` | Connect to a TCP server. |
| `s = tcpserver(address, port)` | Bind + listen; accept one client (lazily, on first I/O). |
| `u = udpport(localPort)` | A connectionless UDP socket bound to `localPort` (`0` = ephemeral). |
| `t.write(data)` | Send a numeric matrix (raw `float64`, one element per 8 bytes). |
| `u.write(data, address, port)` | UDP send to an explicit destination. |
| `r = t.read(count)` | Receive up to `count` `float64` elements (a `1×count` row). |
| `t.writeline(str)` / `s = t.readline()` | Line-oriented text I/O. |
| `t.flush()` | Discard buffered input. |

**Call methods with dot syntax** — `c.write(data)`, `s.read(n)` — the same
convention as the `sim3d` objects (the bare `read(c, n)` form is not dispatched
to a classdef method by this frontend).

## Tier-1 semantics

- **Non-blocking** with a bounded read timeout: a read returns what is available
  (possibly empty) rather than blocking the program forever — an absent peer can
  never deadlock a run.
- **Payloads are raw `float64`.** This is self-consistent for sim↔sim (both ends
  share the runtime). Byte-exact interop with arbitrary third-party tools is a
  later tier; compose framing/typing with an upstream function block.
- **Numeric `write`/`read` have full interpreted↔compiled parity.** Text
  `writeline`/`readline` work in the interpreter; in the compiled lane the
  string method-return typing is a separate gap, so prefer the byte path there.
- **One handle per thread** (the runtime socket state is not thread-safe).

## Examples

- **udp_loopback.m** — a self-contained "sensor → logger" stream over UDP
  loopback (both ends in one program). Runs interpreted and compiled with
  identical output.
- **tcp_sender.m** / **tcp_receiver.m** — the two-process pattern: start the
  receiver in one terminal and the sender in another. They exchange a short
  numeric stream over TCP.

## Two processes talking

```sh
# Terminal 1 — the server/receiver:
matlabc -repl < tcp_receiver.m
# Terminal 2 — the client/sender:
matlabc -repl < tcp_sender.m
```

## Block-diagram counterpart

The mflowlink surface has the matching socket *blocks*
(`signal_udp_send`/`signal_udp_recv`, `signal_tcp_send`/`signal_tcp_recv`; see
[`docs/mflowlink_blocks.md`](../../docs/mflowlink_blocks.md)), including a 3-D
example that streams a `signal_actor3d` transform to an external client
([`examples/mflowlink/3d/net_stream_orbit.mflow`](../mflowlink/3d/net_stream_orbit.mflow)).
