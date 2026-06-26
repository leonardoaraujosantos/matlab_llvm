## ADDED Requirements

### Requirement: TCP client object

The system SHALL provide a `tcpclient(address, port)` handle object that opens a TCP
connection to a server and exposes byte/line read and write methods. The connection
SHALL be backed by C++ runtime socket state keyed by the object handle, created when
the constructor runs and closed when the program ends. Construction failures (refused
connection, bad address) SHALL surface as an error rather than a silent no-op.

#### Scenario: Connect and exchange bytes with a server

- **WHEN** a program runs `t = tcpclient("127.0.0.1", 4000)` against a listening server, then `write(t, [1 2 3 4])` and `r = read(t, 4)`
- **THEN** the four bytes are sent over the socket and `r` is a 1×4 numeric row of the bytes the server returned

#### Scenario: Construction against no server fails loudly

- **WHEN** `tcpclient("127.0.0.1", 4000)` is called and no server is listening on that port
- **THEN** the call raises an error (it does not return a half-open handle that silently drops later writes)

#### Scenario: Read with no data available respects the timeout

- **WHEN** `read(t, 8)` is called but fewer than 8 bytes have arrived within the configured timeout
- **THEN** the call returns the bytes that were available (possibly empty) without blocking the program indefinitely

### Requirement: TCP server object

The system SHALL provide a `tcpserver(address, port)` handle object that binds and
listens on the given port and accepts a single client connection, exposing the same
read/write/line surface as `tcpclient`. This lets one simulation act as the server
endpoint of a two-process link.

#### Scenario: Server accepts a client and echoes data

- **WHEN** one program runs `s = tcpserver("0.0.0.0", 4000)` and a second program connects as a `tcpclient` and writes data
- **THEN** the server program's `read(s, n)` returns the bytes the client wrote, and `write(s, ...)` delivers bytes back to the client

### Requirement: UDP port object

The system SHALL provide a `udpport(...)` handle object for connectionless datagrams,
exposing `write(u, data, address, port)` to send a datagram to a destination and
`read(u, count)` to receive available datagram bytes. Tier 1 binds to an ephemeral or
caller-specified local port.

#### Scenario: Send and receive a datagram on loopback

- **WHEN** two `udpport` handles are bound on `127.0.0.1`, one calls `write(u1, [10 20 30], "127.0.0.1", 5005)`, and the other calls `read(u2, 3)`
- **THEN** the receiver returns the 1×3 row `[10 20 30]`

### Requirement: Line-oriented text I/O

The system SHALL provide `writeline(t, str)` and `readline(t)` on every networking
object. `writeline` SHALL append the line terminator and send the string; `readline`
SHALL return the next terminated line as a string, or an empty string if none is
available within the timeout.

#### Scenario: Round-trip a text line

- **WHEN** one endpoint calls `writeline(a, "hello")` and the peer calls `s = readline(b)`
- **THEN** `s` equals `"hello"` (without the trailing terminator)

### Requirement: Flush buffered data

The system SHALL provide `flush(t)` to discard any buffered input (and, where
applicable, force pending output), so a program can resynchronise a stream.

#### Scenario: Flush drops stale buffered input

- **WHEN** unread bytes are sitting in the input buffer and the program calls `flush(t)` then `read(t, n)`
- **THEN** the post-flush read does not return the discarded pre-flush bytes

### Requirement: Numeric-array and string payloads

`write`/`read` SHALL accept and return numeric matrices (sent as their byte
representation per Tier-1 default datatype), and `writeline`/`readline` SHALL accept
and return strings, using the same `matlab_mat*` / `matlab_string*` marshalling the
runtime already uses for other toolbox objects.

#### Scenario: A numeric vector survives a write/read round-trip

- **WHEN** `write(a, v)` sends a numeric vector `v` and the peer reads the same number of elements back
- **THEN** the received vector equals `v` element-for-element

### Requirement: Interpreted and compiled parity

Networking programs SHALL behave identically in the interpreted REPL (`matlabc -repl`)
and the compiled lane, because both link the same runtime socket implementation.

#### Scenario: Same program, same result in both lanes

- **WHEN** a loopback send/receive program is run via `-repl` and via the compiled binary
- **THEN** both produce the same received data and the same printed output
