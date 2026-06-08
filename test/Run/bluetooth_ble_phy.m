% Bluetooth Toolbox Tier-1 — Bluetooth LE PHY waveform + ideal receiver.
%   bleWaveformGenerator -> (awgn) -> bleIdealReceiver round trip across the
%   four PHY modes.  Zero-noise and high-SNR recovery is exact (BER 0); the
%   coded PHYs (LE500K/LE125K) add FEC coding gain.
rng(1);
dataLen = 200;
bits = round(rand(dataLen, 1));

% Waveform length = (preamble 8 + access addr 32 + PDU + CRC 24) * sps.
wf = bleWaveformGenerator(bits, 'LE1M', 8, 17);
fprintf('LE1M waveform samples: %.0f\n', numel(wf));   % (8+32+200+24)*8 = 2112

% Exact zero-noise round trip for every PHY mode.
fprintf('LE1M   zero-noise BER: %.4f\n', biterr(bits, bleIdealReceiver(bleWaveformGenerator(bits,'LE1M',8,17),   'LE1M',  8, 17)));
fprintf('LE2M   zero-noise BER: %.4f\n', biterr(bits, bleIdealReceiver(bleWaveformGenerator(bits,'LE2M',8,17),   'LE2M',  8, 17)));
fprintf('LE500K zero-noise BER: %.4f\n', biterr(bits, bleIdealReceiver(bleWaveformGenerator(bits,'LE500K',8,17), 'LE500K',8, 17)));
fprintf('LE125K zero-noise BER: %.4f\n', biterr(bits, bleIdealReceiver(bleWaveformGenerator(bits,'LE125K',8,17), 'LE125K',8, 17)));

% High-SNR recovery is also exact.
rxHi = bleIdealReceiver(awgn(bleWaveformGenerator(bits,'LE1M',8,17), 30), 'LE1M', 8, 17);
fprintf('LE1M   30dB BER: %.4f\n', biterr(bits, rxHi));

% Coded-PHY length expansion: rate-1/2 conv (+ K-1 tail) then spreading.
fprintf('LE500K waveform samples: %.0f\n', numel(bleWaveformGenerator(bits,'LE500K',8,17)));
