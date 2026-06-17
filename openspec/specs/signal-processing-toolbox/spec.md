# Signal Processing Toolbox Spec

## Purpose
Documents the shipped function-form subset of the Signal Processing Toolbox in the matlab_llvm compiler: classical IIR/FIR filter design, window generation, frequency-response and filter-application primitives, spectral analysis, FFT-family transforms, multirate resampling, waveform generators, and pulse measurements. The day-to-day design loop (Tiers 1-4) is shipped; elliptic design and System-Object surfaces remain open. (doc: docs/signal_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp) (src: runtime/runtime_complex.cpp)

## Requirements

### Requirement: IIR and FIR filter design with order selection and form conversions
The system SHALL provide Butterworth, Chebyshev type I and type II, and Bessel filter design across lowpass/highpass/bandpass/bandstop, plus order-selection and transfer-function form conversions. (doc: docs/signal_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Design an IIR filter and convert its form
- **WHEN** a program calls `butter`, `cheby1`, `cheby2`, `besself`, `buttord`/`cheb1ord`/`cheb2ord`, `bilinear`, `fir1`, `sgolay`/`sgolayfilt`, or form conversions `tf2zp`/`zp2tf`/`tf2sos`/`sos2tf`
- **THEN** the system SHALL return the corresponding numerator/denominator (or zero-pole-gain / second-order-section) coefficients computed by the runtime design kernels (e.g. `matlab_butter_b`/`matlab_butter_a`, `matlab_cheby1_*`, `matlab_buttord_n`, `matlab_fir1`, `matlab_tf2sos`)

### Requirement: Window functions
The system SHALL provide fixed and parametric window generators. (src: runtime/matlab_runtime.cpp)

#### Scenario: Generate a window vector
- **WHEN** a program calls `hamming`, `hann`, `blackman`, `blackmanharris`, `rectwin`, `triang`, `bartlett`, `barthannwin`, `bohmanwin`, `parzenwin`, `nuttallwin`, `flattopwin`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, or `taylorwin`
- **THEN** the system SHALL return the length-N window column vector computed by the matching `matlab_<window>` runtime entry

### Requirement: Frequency response and filter application
The system SHALL provide frequency-response evaluation and time-domain filtering primitives. (doc: docs/signal_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Evaluate response and filter a signal
- **WHEN** a program calls `freqz`/`freqs`, `filter`, `filtfilt`, `sosfilt`, `impz`, `stepz`, or `grpdelay`
- **THEN** the system SHALL return the digital/analog frequency response or the filtered signal computed via `matlab_freqz`/`matlab_freqs`, direct-form II transposed `matlab_filter`, zero-phase `matlab_filtfilt`, or second-order-section `matlab_sosfilt`

### Requirement: Spectral analysis and FFT-family transforms
The system SHALL provide nonparametric and parametric spectral estimators plus FFT-family transforms. (doc: docs/signal_toolbox_roadmap.md) (src: runtime/runtime_complex.cpp)

#### Scenario: Estimate a spectrum or take a transform
- **WHEN** a program calls `periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate`, `spectrogram`, the AR estimators `levinson`/`lpc`/`aryule`/`arburg`/`pyulear`/`pburg`, or transforms `fft`/`ifft`/`fft2`/`ifft2`/`fftshift`/`ifftshift`/`hilbert`/`dct`/`idct`/`fwht`/`goertzel`
- **THEN** the system SHALL return the power spectral density, time-frequency matrix, AR coefficients, or transformed signal computed by the corresponding runtime entry (e.g. `matlab_pwelch`, `matlab_spectrogram`, `matlab_levinson`, `matlab_fft_c`)

### Requirement: Multirate processing and waveform generation
The system SHALL provide polyphase resampling and analytic waveform generators. (src: runtime/matlab_runtime.cpp)

#### Scenario: Resample a signal or generate a waveform
- **WHEN** a program calls `upfirdn`, `decimate`, `interp`, `resample`, `chirp`, `sawtooth`, `square`, `gauspuls`, `rectpuls`, `tripuls`, or `sinc`
- **THEN** the system SHALL return the resampled signal or generated waveform computed by the matching runtime entry (e.g. `matlab_resample`, `matlab_chirp`)

### Requirement: Convolution, correlation, and pulse measurements
The system SHALL provide convolution/correlation operators and pulse/feature measurement functions. (doc: docs/signal_toolbox_roadmap.md) (src: runtime/matlab_runtime.cpp)

#### Scenario: Measure pulse features or correlate signals
- **WHEN** a program calls `conv`/`conv2`/`xcorr`/`xcov`, peak/feature measurements `findpeaks`/`rms`/`peak2peak`/`peak2rms`/`rssq`/`risetime`/`falltime`/`overshoot`/`undershoot`/`settlingtime`/`dutycycle`/`slewrate`/`statelevels`/`midcross`/`pulseperiod`/`pulsewidth`, denoisers `medfilt1`/`hampel`/`envelope`, or alignment `finddelay`/`dtw`
- **THEN** the system SHALL return the correlation, measured scalar/vector, or aligned result computed by the matching runtime entry (e.g. `matlab_xcorr`, `matlab_findpeaks_pks`/`matlab_findpeaks_locs`, `matlab_finddelay_s`)
