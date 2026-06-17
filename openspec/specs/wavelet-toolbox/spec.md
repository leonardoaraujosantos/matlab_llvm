# Wavelet Toolbox Spec

## Purpose
Documents the shipped function-form subset of the Wavelet Toolbox in the matlab_llvm compiler: 1-D/2-D discrete wavelet transforms with a built-in family-filter catalogue, thresholding and denoising/compression, the continuous wavelet transform and time-frequency analysis, undecimated transforms (SWT/MODWT) and multiresolution analysis, wavelet packets, and special-topic decompositions (EMD/VMD/EWT/OMP/scattering). All six tiers are shipped; there are no classdef files (the surface is function-form in runtime_wavelet.cpp). (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

## Requirements

### Requirement: Discrete wavelet transform and family filters
The system SHALL provide single- and multi-level 1-D discrete wavelet decomposition/reconstruction, coefficient extraction, and a wavelet-family filter catalogue. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Decompose and reconstruct a signal
- **WHEN** a program calls `dwt`/`idwt`, `wavedec`/`waverec`, `appcoef`/`detcoef`, `wfilters`, `wmaxlev`, `wentropy`/`wenergy`, or border helpers `wextend`/`wkeep`, using a family such as `haar`, `db2`-`db9`, `sym2`-`sym8`, or `coif1`-`coif5`
- **THEN** the system SHALL return the approximation/detail coefficients or reconstructed signal via the matching runtime entry (e.g. `matlab_wavelet_dwt_cA`/`matlab_wavelet_dwt_cD`, `matlab_wavelet_wavedec_C`/`matlab_wavelet_wavedec_L`, `matlab_wavelet_waverec`, `matlab_wavelet_wfilters`)

### Requirement: Thresholding, denoising, and compression
The system SHALL provide coefficient thresholding, threshold selection, noise estimation, denoising, and compression. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Denoise a signal
- **WHEN** a program calls `wthresh`, `thselect`, `wnoisest`, `wden`/`wdenoise`, `wcompress`, or `measerr`
- **THEN** the system SHALL return the thresholded coefficients, selected threshold, estimated noise level, denoised signal, or error metrics via the matching runtime entry (e.g. `matlab_wavelet_wthresh`, `matlab_wavelet_thselect`, `matlab_wavelet_wnoisest1`, `matlab_wavelet_wdenoise2`)

### Requirement: Continuous wavelet transform and time-frequency analysis
The system SHALL provide the continuous wavelet transform, its inverse, scale/frequency conversion, and wavelet coherence. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Compute a scalogram
- **WHEN** a program calls `cwt`/`icwt`, `scal2frq`/`freq2scal`, or `wcoherence` using an analytic family such as `amor`, `morse`, or `bump`
- **THEN** the system SHALL return the CWT coefficient magnitude and frequencies, the reconstructed signal, the converted scale/frequency, or the coherence map via the matching runtime entry (e.g. `matlab_wavelet_cwt_mag`/`matlab_wavelet_cwt_f`, `matlab_wavelet_icwt`, `matlab_wavelet_wcoherence`)

### Requirement: Undecimated transforms, MODWT, and 2-D DWT
The system SHALL provide stationary/MODWT transforms with multiresolution analysis and 2-D discrete wavelet transforms. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Run MODWT MRA or a 2-D transform
- **WHEN** a program calls `swt`/`iswt`, `modwt`/`imodwt`/`modwtmra`/`modwtvar`, or 2-D `dwt2`/`idwt2`/`wavedec2`/`waverec2`/`wcodemat`
- **THEN** the system SHALL return the undecimated coefficients, MRA components, scale-localized variance, or 2-D subbands via the matching runtime entry (e.g. `matlab_wavelet_swt`, `matlab_wavelet_modwt2`/`matlab_wavelet_imodwt1`/`matlab_wavelet_modwtmra1`, `matlab_wavelet_dwt2_cA`/`matlab_wavelet_dwt2_cH`/`matlab_wavelet_dwt2_cV`/`matlab_wavelet_dwt2_cD`)

### Requirement: Wavelet packets
The system SHALL provide wavelet-packet decomposition, reconstruction, coefficient extraction, and best-basis selection. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Build a best-basis packet tree
- **WHEN** a program calls `wpdec`, `wprec`, `wpcoef`, or `besttree`
- **THEN** the system SHALL return the packet tree, reconstruction, node coefficients, or entropy-optimal basis via the matching runtime entry (e.g. `matlab_wavelet_wpdec`, `matlab_wavelet_wprec`, `matlab_wavelet_wpcoef`, `matlab_wavelet_besttree`)

### Requirement: Adaptive decompositions and feature extraction
The system SHALL provide empirical/variational mode decompositions, the empirical wavelet transform, orthogonal matching pursuit, and wavelet scattering. (doc: docs/wavelet_toolbox_roadmap.md) (src: runtime/toolbox/wavelet/runtime_wavelet.cpp)

#### Scenario: Decompose with an adaptive method or extract features
- **WHEN** a program calls `emd`, `vmd`, `ewt`, `omp`, or wavelet `scattering`
- **THEN** the system SHALL return the intrinsic modes, sparse coefficients, or scattering feature matrix via the matching runtime entry (e.g. `matlab_wavelet_emd`, `matlab_wavelet_vmd`, `matlab_wavelet_ewt`, `matlab_wavelet_omp`, `matlab_wavelet_scatter`)
