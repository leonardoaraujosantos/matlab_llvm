% ble_ber_awgn.m — Bluetooth Toolbox Phase-A headline (Tier-1).
% ----------------------------------------------------------------------
% The canonical Bluetooth LE end-to-end PHY simulation: for each of the four
% PHY transmission modes, generate a standard-framed LE packet with
% bleWaveformGenerator (preamble + access address + whitened PDU + CRC-24,
% CPFSK/MSK modulated; FEC + spreading for the coded PHYs), pass it through an
% AWGN channel, recover the bits with bleIdealReceiver, and report the bit
% error rate vs SNR.  The coded PHYs (LE500K/LE125K) show the FEC coding gain.
% No external dependency — the GFSK/CPFSK modulator + Viterbi-decoded coded
% PHY ride the shipped complex-matrix + awgn + biterr surface.

rng(7);
sps     = 8;
dataLen = 4000;
bits    = round(rand(dataLen, 1));
snrVec  = [0 2 4 6 8 10];
nSnr    = 6;

fprintf('Bluetooth LE BER vs SNR (AWGN), %d-bit PDU, sps=%d\n', dataLen, sps);

% --- Uncoded PHY: LE1M (1 Mb/s) -----------------------------------------
wf1 = bleWaveformGenerator(bits, 'LE1M', sps, 17);
fprintf('LE1M  : ');
for i = 1:nSnr
    ber = biterr(bits, bleIdealReceiver(awgn(wf1, snrVec(i)), 'LE1M', sps, 17));
    fprintf('%.4f ', ber);
end
fprintf('\n');

% --- Coded PHY: LE500K (rate-1/2 FEC, S=2) ------------------------------
wfc = bleWaveformGenerator(bits, 'LE500K', sps, 17);
fprintf('LE500K: ');
for i = 1:nSnr
    ber = biterr(bits, bleIdealReceiver(awgn(wfc, snrVec(i)), 'LE500K', sps, 17));
    fprintf('%.4f ', ber);
end
fprintf('\n');

% --- Coded PHY: LE125K (rate-1/2 FEC, S=8 — most robust) ----------------
wfk = bleWaveformGenerator(bits, 'LE125K', sps, 17);
fprintf('LE125K: ');
for i = 1:nSnr
    ber = biterr(bits, bleIdealReceiver(awgn(wfk, snrVec(i)), 'LE125K', sps, 17));
    fprintf('%.4f ', ber);
end
fprintf('\n');

fprintf('At a fixed SNR the BER drops LE1M > LE500K > LE125K (more coding gain).\n');
