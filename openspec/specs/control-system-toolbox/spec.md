# Control System Toolbox Spec

## Purpose
Documents the shipped subset of the Control System Toolbox in the matlab_llvm compiler: LTI model objects (`tf`/`ss`/`zpk`/`frd`/`pid`) with operator overloads, continuous-to-discrete conversions and data unpackers, time- and frequency-domain analysis, pole/interconnection operations, LQ/pole-placement design, gramians, and model reduction. Tier-1 numerics and most of Tier-2 are shipped; some MIMO and delay features remain deferred. (doc: docs/control_toolbox_roadmap.md) (src: runtime/toolbox/control) (src: runtime/matlab_runtime.cpp)

## Requirements

### Requirement: LTI model objects with operator overloads
The system SHALL provide `tf`, `ss`, `zpk`, `frd`, and `pid` model objects with constructors and arithmetic operator overloads. (src: runtime/toolbox/control/cst_classdefs.m) (src: runtime/toolbox/control/cst_class_ss.m) (src: runtime/toolbox/control/cst_class_zpk.m) (src: runtime/toolbox/control/cst_class_frd.m) (src: runtime/toolbox/control/cst_class_pid.m)

#### Scenario: Build and combine LTI models
- **WHEN** a program constructs a `tf` (including `tf('s')`/`tf('z')` sugar), `ss`, `zpk`, `frd`, or `pid` and combines models with `+`, `-`, `*`, `/`, or unary minus
- **THEN** the system SHALL return a model object holding the corresponding parameters (num/den, A/B/C/D/Ts, Z/P/K, ResponseData/Frequency, or Kp/Ki/Kd/Tf) with operator results computed by the classdef overloads

### Requirement: Discretization and model-data unpacking
The system SHALL provide continuous-to-discrete and discrete-to-continuous conversion (ZOH and Tustin) and model-data unpackers. (doc: docs/control_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Discretize a plant and unpack its data
- **WHEN** a program calls `c2d`/`d2c` (ZOH or Tustin, matrix- or tf-form) or unpacks a model with `ssdata`/`tfdata`
- **THEN** the system SHALL return the discretized matrices/coefficients via `matlab_c2d_Ad`/`matlab_c2d_Bd`, `matlab_c2d_tustin_*`, `matlab_c2d_tf_*`, or `matlab_d2c_*`, and the `[A,B,C,D]`/`[num,den]` data via the classdef `ssdata`/`tfdata` methods

### Requirement: Time- and frequency-domain analysis
The system SHALL provide time-response and frequency-response analysis functions. (doc: docs/control_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Analyze a system response
- **WHEN** a program calls `step`, `impulse`, `lsim`, `initial`, `stepinfo`, `bode`, `freqresp`, `nyquist`, `margin`/`allmargin`, `dcgain`, `bandwidth`, or `damp`
- **THEN** the system SHALL return the response trajectory, magnitude/phase, gain/phase margins, or scalar metric computed by the matching runtime entry (e.g. `matlab_step_ss`, `matlab_bode_ss_mag`/`matlab_bode_ss_phase`, `matlab_gain_margin`/`matlab_phase_margin`, `matlab_dcgain_ss`)

#### Scenario: Data-returning multi-output response forms
- **WHEN** a program assigns the response data to multiple outputs — `[y,t] = step(sys)`, `[y,t] = impulse(sys)`, `[y,t] = initial(sys,x0)`, `[mag,phase,w] = bode(sys)` (auto frequency grid) or `bode(sys,w)`, `[re,im,w] = nyquist(sys)` or `nyquist(sys,w)`
- **THEN** the system SHALL return the response/grid data instead of plotting, splitting the model object's matrices to the per-output runtime entries; auto-grid forms synthesise a default grid (time: dt=0.01, N=500; frequency: logspace(-2,3,200)) (src: lib/MLIR/Lowering.cpp model-object multi-return splitters; test: test/Run/ctrl_step_data.m, test/Run/ctrl_response_data.m)

### Requirement: Pole analysis and model interconnections
The system SHALL provide pole/stability analysis and SISO model interconnection operators. (doc: docs/control_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Compute poles or interconnect models
- **WHEN** a program calls `pole`, `isstable`, or interconnections `feedback`/`series`/`parallel`/`append`
- **THEN** the system SHALL return the system poles (eigenvalues of A), a stability flag, or a fresh interconnected `ss` model computed by `matlab_pole`/`matlab_isstable` or `matlab_feedback_ss_*`/`matlab_series_ss_*`/`matlab_parallel_ss_*`/`matlab_append_ss_*`

### Requirement: Optimal control design, gramians, and Riccati solvers
The system SHALL provide LQ/pole-placement design, controllability/observability gramians, and Riccati/Lyapunov solvers. (doc: docs/control_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Design a controller or estimator
- **WHEN** a program calls `lqr`/`dlqr` (1- or 3-return, 5-arg cross-term), `place`/`acker`, `kalman`, `ctrb`/`obsv`/`gram`, or `care`/`dare`/`lyap`/`dlyap`/`sylvester`
- **THEN** the system SHALL return the gain/estimator/gramian/Riccati solution computed by the matching runtime entry (e.g. `matlab_lqr`, `matlab_place`, `matlab_kalman_L`, `matlab_ctrb`/`matlab_obsv`/`matlab_gram_c`/`matlab_gram_o`, `matlab_care`/`matlab_dare`)

### Requirement: Model reduction and norms
The system SHALL provide balanced realization, balanced reduction, minimal realization, and H2 norm computation. (doc: docs/control_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Reduce a model or compute its norm
- **WHEN** a program calls `balreal`, `balred`, `hsvd`, `minreal` (tf-form), `sminreal`, `modred`, or `norm`/`norm(sys,2)`
- **THEN** the system SHALL return the transformed/reduced model, Hankel singular values, or H2 norm computed by the matching runtime entry (e.g. `matlab_balreal_T`, `matlab_balred_A`/`matlab_balred_B`/`matlab_balred_C`, `matlab_hsvd`, `matlab_minreal_tf_num`/`matlab_minreal_tf_den`, `matlab_norm_h2`)
