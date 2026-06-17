# Bluetooth Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Bluetooth Toolbox in `matlab_llvm`: BLE/BR PHY waveform generation and reception, link-layer/L2CAP framing, channel-selection and frequency utilities, and angle-of-arrival localization. Layered on the shipped Communications + DSP stack.

## Requirements

### Requirement: BLE PHY waveform generation
The system SHALL generate Bluetooth Low Energy / BR GFSK waveforms.

#### Scenario: Generate a BLE waveform
- **WHEN** a program calls `bleWaveformGenerator`-style waveform generation or the generic Bluetooth waveform generator
- **THEN** the system SHALL return the modulated baseband samples (matlab_bluetooth_ble_wavegen, matlab_bluetooth_wavegen) (doc: docs/bluetooth_toolbox_roadmap.md) (src: runtime/toolbox/bluetooth/runtime_bluetooth.cpp)

### Requirement: PHY reception and recovery
The system SHALL receive and demodulate Bluetooth waveforms.

#### Scenario: Receive a BLE waveform
- **WHEN** a program calls the BLE receiver or generic Bluetooth receiver, applying frequency-offset correction
- **THEN** the system SHALL return the recovered bits and PHY metrics (matlab_bluetooth_ble_rx, matlab_bluetooth_rx, matlab_bluetooth_freqoffset, matlab_bluetooth_freqdev) (doc: docs/bluetooth_toolbox_roadmap.md) (src: runtime/toolbox/bluetooth/runtime_bluetooth.cpp)

### Requirement: Link-layer and L2CAP framing
The system SHALL encode and decode link-layer PDUs and L2CAP frames.

#### Scenario: Encode and decode a PDU
- **WHEN** a program builds a link-layer PDU or L2CAP frame and decodes the received bytes
- **THEN** the system SHALL return the encoded frame or decoded fields (matlab_bluetooth_ll_pdu, matlab_bluetooth_ll_pdu_decode, matlab_bluetooth_l2cap, matlab_bluetooth_l2cap_decode) (doc: docs/bluetooth_toolbox_roadmap.md) (src: runtime/toolbox/bluetooth/runtime_bluetooth.cpp)

### Requirement: Channel selection and frequency mapping
The system SHALL map Bluetooth channels to frequencies and run channel-selection algorithms.

#### Scenario: Select a channel
- **WHEN** a program calls channel-selection (`chsel`) or channel-to-frequency mapping (`ch2freq`)
- **THEN** the system SHALL return the selected channel index or center frequency (matlab_bluetooth_chsel, matlab_bluetooth_ch2freq) (doc: docs/bluetooth_toolbox_roadmap.md) (src: runtime/toolbox/bluetooth/runtime_bluetooth.cpp)

### Requirement: Angle-of-arrival localization
The system SHALL estimate angle of arrival for Bluetooth direction finding.

#### Scenario: Estimate angle of arrival
- **WHEN** a program calls the angle-of-arrival estimator on IQ samples from an antenna array
- **THEN** the system SHALL return the estimated angle(s) of arrival (matlab_bluetooth_aoa) (doc: docs/bluetooth_toolbox_roadmap.md) (src: runtime/toolbox/bluetooth/runtime_bluetooth.cpp)
