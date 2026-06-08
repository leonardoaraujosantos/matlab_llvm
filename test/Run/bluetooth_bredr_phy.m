% Bluetooth Toolbox Tier-2 — Bluetooth BR/EDR PHY.
%   bluetoothWaveformGenerator -> (awgn) -> bluetoothIdealReceiver across the
%   classic-Bluetooth modes: BR (GFSK 1 Mb/s), EDR2M (pi/4-DQPSK 2 Mb/s),
%   EDR3M (8-DPSK 3 Mb/s).  Zero-noise + high-SNR recovery is exact.
rng(5);
dataLen = 196;                 % aligned for 1/2/3 bits-per-symbol framing
bits = round(rand(dataLen, 1));

wbr = bluetoothWaveformGenerator(bits, 'BR', 8);
fprintf('BR    waveform samples: %.0f\n', numel(wbr));   % (8+196+24)*8 = 1840
fprintf('BR    zero-noise BER: %.4f\n', biterr(bits, bluetoothIdealReceiver(wbr, 'BR', 8)));

w2 = bluetoothWaveformGenerator(bits, 'EDR2M', 8);
fprintf('EDR2M waveform samples: %.0f\n', numel(w2));     % 228/2 * 8 = 912
fprintf('EDR2M zero-noise BER: %.4f\n', biterr(bits, bluetoothIdealReceiver(w2, 'EDR2M', 8)));

w3 = bluetoothWaveformGenerator(bits, 'EDR3M', 8);
fprintf('EDR3M waveform samples: %.0f\n', numel(w3));     % 228/3 * 8 = 608
fprintf('EDR3M zero-noise BER: %.4f\n', biterr(bits, bluetoothIdealReceiver(w3, 'EDR3M', 8)));

fprintf('BR    25dB BER: %.4f\n', biterr(bits, bluetoothIdealReceiver(awgn(wbr, 25), 'BR', 8)));
fprintf('EDR2M 25dB BER: %.4f\n', biterr(bits, bluetoothIdealReceiver(awgn(w2, 25), 'EDR2M', 8)));
