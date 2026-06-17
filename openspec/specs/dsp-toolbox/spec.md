# DSP System Toolbox Spec

## Purpose
Documents the shipped subset of the DSP System Toolbox in the matlab_llvm compiler: streaming System objects (`dsp.*`) for filtering, adaptive filtering, multirate, signal generation, moving statistics, and detection, each backed by a stateful runtime step kernel, plus a hardware-oriented `dsphdl.*` mirror and equiripple/notch design helpers. (doc: docs/dsp_toolbox_roadmap.md) (src: runtime/toolbox/dsp)

## Requirements

### Requirement: Filter System objects
The system SHALL provide stateful filter System objects with step/reset lifecycle. (src: runtime/toolbox/dsp/dsp_classdefs.m) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Step a filter object
- **WHEN** a program constructs `dsp.FIRFilter`, `dsp.IIRFilter`, `dsp.SOSFilter`, `dsp.BiquadFilter`, `dsp.LowpassFilter`, `dsp.HighpassFilter`, `dsp.NotchPeakFilter`, or `dsp.Delay` and calls it on a frame
- **THEN** the system SHALL return the filtered frame computed by the matching runtime step (e.g. `matlab_dsp_iir_step`, `matlab_dsp_sos_step`, `matlab_dsp_delay_step`) with discrete state persisted across calls and cleared by `reset`

### Requirement: Adaptive filter System objects
The system SHALL provide LMS and RLS adaptive filter System objects. (src: runtime/toolbox/dsp/dsp_classdefs.m) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Adapt and read weights
- **WHEN** a program constructs `dsp.LMSFilter` or `dsp.RLSFilter`, steps it with input and desired signals, and reads its weights
- **THEN** the system SHALL return the filtered output and updated tap weights via `matlab_dsp_lms_step`/`matlab_dsp_rls_step`, with weights retrievable through `matlab_dsp_get_weights`

### Requirement: Multirate System objects
The system SHALL provide FIR/CIC decimation and interpolation plus rational sample-rate conversion System objects. (src: runtime/toolbox/dsp/dsp_classdefs.m) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Change sample rate
- **WHEN** a program constructs `dsp.FIRDecimator`, `dsp.FIRInterpolator`, `dsp.CICDecimator`, `dsp.CICInterpolator`, or `dsp.SampleRateConverter` and steps it
- **THEN** the system SHALL return the rate-converted frame computed by `matlab_dsp_firdecim_step`, `matlab_dsp_firinterp_step`, `matlab_dsp_cicdecim_step`, `matlab_dsp_cicinterp_step`, or `matlab_dsp_rateconv_step`

### Requirement: Sources, moving statistics, detection, and buffering System objects
The system SHALL provide signal-source generators, streaming moving statistics, detectors, and buffer System objects. (src: runtime/toolbox/dsp/dsp_classdefs.m) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Generate, measure, detect, or buffer a stream
- **WHEN** a program constructs `dsp.SineWave`, `dsp.NCO`, `dsp.Chirp`, `dsp.MovingAverage`/`dsp.MovingRMS`/`dsp.MovingMaximum`/`dsp.MovingMinimum`/`dsp.MovingStandardDeviation`, `dsp.PeakFinder`, `dsp.DCBlocker`, `dsp.ZeroCrossingDetector`, `dsp.SpectrumEstimator`, `dsp.LevinsonSolver`, or `dsp.AsyncBuffer`
- **THEN** the system SHALL return the generated/measured/detected/buffered frame computed by the matching runtime step (e.g. `matlab_dsp_sine_step`, `matlab_dsp_movavg_step`, `matlab_dsp_zcd_step`, `matlab_dsp_asyncbuf_write`/`matlab_dsp_asyncbuf_read`)

### Requirement: Filter design and CORDIC helpers
The system SHALL provide equiripple/least-squares FIR design, notch/peak IIR design, and CORDIC primitives. (doc: docs/dsp_toolbox_roadmap.md) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Design a filter or compute a CORDIC primitive
- **WHEN** a program calls Parks-McClellan/least-squares FIR design, notch/peak design, or CORDIC primitives
- **THEN** the system SHALL return the coefficients or computed value via `matlab_dsp_firpm`, `matlab_dsp_firls`, `matlab_dsp_iirnotch_b`/`matlab_dsp_iirnotch_a`, `matlab_dsp_iirpeak_b`/`matlab_dsp_iirpeak_a`, `matlab_dsp_cordic_atan2`, or `matlab_dsp_cordic_sqrt`

### Requirement: DSP HDL System object mirror
The system SHALL provide a hardware-oriented `dsphdl.*` System object mirror that reports pipeline latency. (src: runtime/toolbox/dsp/dsphdl_classdefs.m) (src: runtime/toolbox/dsp/runtime_dsp.cpp)

#### Scenario: Step an HDL-targeted object
- **WHEN** a program constructs `dsphdl.FIRFilter`, `dsphdl.BiquadFilter`, `dsphdl.SineWave`, `dsphdl.NCO`, `dsphdl.FIRDecimator`, or `dsphdl.CICDecimator`
- **THEN** the system SHALL return the bit-true output via the matching `matlab_dsphdl_*_step` wrapper and report pipeline latency through `matlab_dsphdl_latency`
