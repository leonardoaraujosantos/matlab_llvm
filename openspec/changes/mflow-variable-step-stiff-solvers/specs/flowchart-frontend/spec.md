## MODIFIED Requirements

### Requirement: mflowlink block model and simulation
The system SHALL build an mflowlink signal-flow block model from a `.mflow` document, classify block sample times, break algebraic loops, and run an in-process simulator. The simulator SHALL honour the document's `settings.solver` selection — `type` (`fixed_step` / `variable_step`), `algorithm`, tolerances (`relTol` / `absTol`), `maxStep` / `minStep`, and `refine` — choosing a fixed-step, variable-step explicit, or stiff implicit integrator accordingly. (src: include/matlab/Flowchart/MflowLinkModel.h, src: lib/Flowchart/MflowLinkSim.cpp, doc: docs/mflowlink_blocks.md)

#### Scenario: Sample-time classification and loop breaking
- **WHEN** a signal-flow model is built
- **THEN** the system SHALL classify each block's sample time (Continuous, Discrete, Constant, FixedInMinor), track continuous/discrete state counts, and mark loop-breaker blocks (Integrator, Unit Delay, strictly-proper transfer functions) so their edges are dropped from the execution-order topological sort (src: include/matlab/Flowchart/MflowLinkModel.h)

#### Scenario: In-process simulation
- **WHEN** an mflowlink model is simulated
- **THEN** the system SHALL step blocks in execution order with continuous-state integration and produce logged signal output (src: lib/Flowchart/MflowLinkSim.cpp, examples/mflowlink/bouncing_ball.mflow, test/Flowchart/Simulate)

#### Scenario: Integrator external reset port
- **WHEN** a `signal_integrator` block has a connected `reset` input and that signal makes a rising edge (`prev ≤ 0 && now > 0`)
- **THEN** the system SHALL reload the integrator's continuous state at the next major step from the `init` input port if connected, else from `initialCondition` (the zero-crossing → state-reset pattern) (src: lib/Flowchart/MflowLinkSim.cpp, examples/mflowlink/bouncing_ball.mflow, test/Flowchart/SimulateRun)

#### Scenario: Per-output-port routing for multi-output blocks
- **WHEN** a block exposes multiple output ports (e.g. a `signal_state_space` with a multi-row `C`, or a MATLAB Function block returning `[y1, y2, …]`) and a consumer is wired from a specific source port (`out2`, …)
- **THEN** the system SHALL route the value published for that source port (each output port `outK` carries its own value), not the block's primary scalar output; a `signal_state_space` with per-state `x0` initial conditions SHALL seed each state independently, and `outK = (C·x)_k` (src: lib/Flowchart/MflowLinkSim.cpp `PortOut_`, test/Flowchart/SimulateRun, examples/mflowlink/state_space_vector_ic.mflow)

#### Scenario: Solver selection drives the integrator
- **WHEN** a model sets `settings.solver.type` and `algorithm`
- **THEN** the simulator SHALL select a fixed-step method (`ode1`/`ode2`/`ode4`), a variable-step explicit method (`ode45`, `ode23`), or a stiff implicit method (`ode15s`, `ode23s`, `ode23t`, `ode23tb`) accordingly, applying adaptive step-size control from `relTol`/`absTol`/`maxStep`/`minStep` for the variable-step and stiff lanes (src: lib/Flowchart/MflowLinkSim.cpp, capability mflow-variable-step-stiff-solvers)

#### Scenario: Solver behaviour is identical in interpreter and compiled binary
- **WHEN** a model is run via `matlabc -simulate` and via the standalone `matlabc -emit-mflowlink-cpp` binary
- **THEN** the chosen solver SHALL produce byte-identical logged output in both, since the compiled binary links the same `MflowLinkSim` evaluator (src: lib/Flowchart/MflowLinkSim.cpp, test/Flowchart/EmitMflowLinkCpp)
