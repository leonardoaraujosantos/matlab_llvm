## ADDED Requirements

### Requirement: Socket block kinds in the catalogue

The system SHALL register four new mflowlink block kinds — `signal_udp_send`,
`signal_udp_recv`, `signal_tcp_send`, `signal_tcp_recv` — in the authoritative block
catalogue (`kindTable`) as Known and Supported, and SHALL list them in the
registered-block-kinds parity snapshot. The `.mflow` loader SHALL accept them without
schema changes (block registration is additive).

#### Scenario: A diagram using a socket block loads and simulates

- **WHEN** a `.mflow` diagram containing a `signal_udp_send` node is run with `matlabc -simulate`
- **THEN** the loader recognises the kind, the engine evaluates it each step, and the run completes without an "unknown block kind" error

#### Scenario: Parity snapshot includes the new kinds

- **WHEN** the block-kind parity test compares the registered kinds against the snapshot
- **THEN** `signal_udp_send`, `signal_udp_recv`, `signal_tcp_send`, and `signal_tcp_recv` are present in the snapshot

### Requirement: Send blocks transmit the input wire at major steps

A send block (`signal_udp_send` / `signal_tcp_send`) SHALL read its input port each
major (output) step and transmit that scalar/vector value to the configured endpoint
(`host`/`port` params). It SHALL perform socket I/O ONLY at major-step boundaries, never
inside an RK4 minor stage, so the continuous integration remains deterministic.

#### Scenario: Each logged step emits one datagram

- **WHEN** a `signal_udp_send` is driven by a source over an N-major-step run
- **THEN** the peer receives N messages, one per major step, carrying the source's value at each step

#### Scenario: Minor stages do not transmit

- **WHEN** the solver takes multiple RK4 minor stages within one major step
- **THEN** the send block transmits exactly once for that major step (not once per minor stage)

### Requirement: Receive blocks are loop-breakers that hold last value

A receive block (`signal_udp_recv` / `signal_tcp_recv`) SHALL be a loop-breaker: its
output for the current step SHALL NOT depend on the current step's input, so it can sit
inside a feedback path without creating an algebraic loop. When no new data has arrived
by the (bounded, non-blocking) drain point, the block SHALL output its last received
value (its `initialValue` param before the first packet) rather than block the solver.

#### Scenario: Output drives a feedback path without an algebraic-loop error

- **WHEN** a `signal_tcp_recv` output feeds a controller whose result loops back through the diagram
- **THEN** the diagram simulates without an algebraic-loop / loop-breaker error

#### Scenario: Starved receiver holds the last value

- **WHEN** the peer sends nothing during several major steps
- **THEN** the receive block keeps emitting the most recent received value (or its initial value) and the run continues at its normal rate

### Requirement: Socket lifecycle bound to the simulation run

Each socket block SHALL open its socket when the simulation resets/starts and close it
when the simulation is destroyed, with per-block socket state held on the simulation
object (not in global state), so repeated runs and multiple blocks each get independent
endpoints.

#### Scenario: Re-running a diagram reopens clean sockets

- **WHEN** the same diagram is simulated twice in one process
- **THEN** the second run opens fresh sockets and does not inherit buffered data from the first

### Requirement: 3-D environment streaming

The system SHALL allow a `signal_actor3d` transform wire (its 9-element
`[tx ty tz rx ry rz sx sy sz]`) to be connected directly into a `signal_udp_send` or
`signal_tcp_send` input, so an mflowlink 3-D environment streams its live actor
transforms to an external client. This path MUST require no new viewer machinery — it
reuses the existing transform wire and the send block.

#### Scenario: A 3-D scene streams its actor transforms

- **WHEN** a `signal_actor3d` transform wire is connected to a `signal_udp_send` and the diagram is simulated
- **THEN** an external UDP client receives the `[tx ty tz rx ry rz sx sy sz]` transform stream, one frame per major step

### Requirement: Deterministic loopback testing

The system SHALL ship at least one paired send/receive example and a loopback-based
regression test in which both endpoints bind `127.0.0.1` with fixed payloads and the
receiver drains at major steps, so the test is reproducible despite using real sockets.

#### Scenario: Loopback regression is reproducible

- **WHEN** the loopback send→recv fixture is run repeatedly
- **THEN** it produces the same received values each time and passes deterministically
