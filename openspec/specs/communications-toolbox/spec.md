# Communications Toolbox Spec

## Purpose
Documents the shipped function-form subset of the Communications Toolbox in the matlab_llvm compiler: random sources and RNG control, bit/integer conversion, digital modulation (PAM/QAM/PSK/FSK), AWGN channel and error-rate measurement, channel coding (CRC/convolutional+Viterbi/Hamming/interleavers/LDPC/Turbo/Polar), equalization and synchronization, RF impairments, OFDM, MIMO and fading channels, spreading sequences, and source coding, plus CRC generator/detector System objects. (doc: docs/comm_toolbox_roadmap.md) (src: runtime/toolbox/comm)

## Requirements

### Requirement: Random sources, RNG control, and bit/integer conversion
The system SHALL provide random integer/alphabet sources, RNG seed control, and bit/integer conversions. (src: runtime/toolbox/comm/runtime_comm.cpp)

#### Scenario: Generate symbols and convert representations
- **WHEN** a program calls `randi`, `rng`, `randsrc`, `randerr`, `int2bit`/`bit2int`, or `de2bi`/`bi2de`
- **THEN** the system SHALL return the random source or converted representation via the matching runtime entry (e.g. `matlab_comm_randi_range`, `matlab_comm_rng`, `matlab_comm_int2bit`/`matlab_comm_bit2int`)

### Requirement: Digital modulation and pulse shaping
The system SHALL provide PAM/QAM/PSK/FSK and generic-constellation modulation/demodulation plus pulse-shaping design and BER reference curves. (doc: docs/comm_toolbox_roadmap.md) (src: runtime/toolbox/comm/runtime_comm.cpp)

#### Scenario: Modulate and demodulate a symbol stream
- **WHEN** a program calls `pammod`/`pamdemod`, `qammod`/`qamdemod` (hard/bit/LLR), `pskmod`/`pskdemod`, `fskmod`/`fskdemod`, `genqammod`/`genqamdemod`, `rcosdesign`/`gaussdesign`, `berawgn`, `scatterplot`, or `eyediagram`
- **THEN** the system SHALL return the modulated/demodulated stream, filter taps, reference BER, or plotted data via the matching runtime entry (e.g. `matlab_comm_qammod`/`matlab_comm_qamdemod_llr`, `matlab_comm_pskmod`, `matlab_comm_rcosdesign`, `matlab_comm_berawgn_s`)

### Requirement: AWGN channel and error-rate measurement
The system SHALL provide an AWGN channel and bit/symbol error-rate measurement. (src: runtime/toolbox/comm/runtime_comm.cpp)

#### Scenario: Add noise and measure errors
- **WHEN** a program calls `awgn`, `biterr`, `symerr`, `qfunc`, or `erfc`
- **THEN** the system SHALL return the noisy signal or error counts/ratios via `matlab_comm_awgn`, `matlab_comm_biterr_count`/`matlab_comm_biterr_ratio`, `matlab_comm_symerr_count`/`matlab_comm_symerr_ratio`, or `matlab_comm_qfunc_s`

### Requirement: Channel coding (classical and modern)
The system SHALL provide CRC, convolutional+Viterbi, Hamming, interleaving, and modern (LDPC/Turbo/Polar) coding in function form, plus CRC generator/detector System objects. (doc: docs/comm_toolbox_roadmap.md) (src: runtime/toolbox/comm/runtime_comm.cpp) (src: runtime/toolbox/comm/comm_class_crc_generator.m) (src: runtime/toolbox/comm/comm_class_crc_detector.m)

#### Scenario: Encode and decode a message
- **WHEN** a program calls `crc` (function-form or `CommCRCGenerator`/`CommCRCDetector` objects), `convenc`/`vitdec` (hard or soft), `hammgen`/Hamming encode-decode, `intrlv`/`deintrlv`, or LDPC/Turbo/Polar encode-decode
- **THEN** the system SHALL return the encoded/decoded bits via the matching runtime entry (e.g. `matlab_comm_crc_generate`/`matlab_comm_crc_check`, `matlab_comm_convenc`/`matlab_comm_vitdec`/`matlab_comm_vitdec_soft`, `matlab_comm_ldpc_encode`/`matlab_comm_ldpc_decode_ms`, `matlab_comm_turbo_encode`/`matlab_comm_turbo_decode`, `matlab_comm_polar_encode`/`matlab_comm_polar_sc_decode`)

### Requirement: Equalization, synchronization, and RF impairments
The system SHALL provide adaptive equalizers, carrier/symbol synchronization, and RF impairment models. (doc: docs/comm_toolbox_roadmap.md) (src: runtime/toolbox/comm/runtime_comm.cpp)

#### Scenario: Equalize, synchronize, or impair a signal
- **WHEN** a program calls `lms`/`rls`/`cma`/`dfe` equalizers, Costas-PLL / Mueller-Müller / preamble-detect synchronizers, or impairment models (phase/frequency offset, phase noise, IQ imbalance, PA nonlinearity)
- **THEN** the system SHALL return the equalized/synchronized/impaired signal via the matching runtime entry (e.g. `matlab_comm_lms`/`matlab_comm_dfe`, `matlab_comm_costas_pll`, `matlab_comm_symbol_sync_mm`, `matlab_comm_iqimbal`, `matlab_comm_memoryless_nl`)

### Requirement: OFDM, MIMO, fading, spreading, and source coding
The system SHALL provide OFDM modulation, MIMO (STBC/ML detection), fading channels, spreading sequences, and source coding. (doc: docs/comm_toolbox_roadmap.md) (src: runtime/toolbox/comm/runtime_comm.cpp)

#### Scenario: Transmit over a fading MIMO channel and quantize a source
- **WHEN** a program calls `ofdmmod`/`ofdmdemod`, `ostbc` encode/combine, ML detection, Rayleigh/Rician channels, PN/Gold/Hadamard/Walsh sequences, or source coding (quantization, Lloyd's design, mu-law/A-law companding, DPCM)
- **THEN** the system SHALL return the corresponding result via the matching runtime entry (e.g. `matlab_comm_ofdmmod`/`matlab_comm_ofdmdemod`, `matlab_comm_ostbc_encode`/`matlab_comm_ostbc_combine`, `matlab_comm_rayleigh_channel`/`matlab_comm_rician_channel`, `matlab_comm_pn_sequence`/`matlab_comm_gold_sequence`, `matlab_comm_quantiz`/`matlab_comm_compand_mu`/`matlab_comm_dpcm_encode`)
