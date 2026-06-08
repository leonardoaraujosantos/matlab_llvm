% bredr_ber.m — Bluetooth Toolbox Phase-A (Tier-2).
% ----------------------------------------------------------------------
% Classic-Bluetooth (BR/EDR) PHY BER vs SNR.  Basic Rate uses GFSK (1 Mb/s);
% Enhanced Data Rate uses differential PSK — pi/4-DQPSK at 2 Mb/s and 8-DPSK
% at 3 Mb/s.  Higher-order DPSK packs more bits per symbol but is less robust
% to noise, so at a fixed SNR the BER rises BR < EDR2M < EDR3M.

rng(11);
sps     = 8;
dataLen = 1200;
bits    = round(rand(dataLen, 1));
snrVec  = [4 8 12 16 20];
nSnr    = 5;

fprintf('Bluetooth BR/EDR BER vs SNR (AWGN), %d-bit payload\n', dataLen);

wbr = bluetoothWaveformGenerator(bits, 'BR', sps);
fprintf('BR    (GFSK)     : ');
for i = 1:nSnr
    fprintf('%.4f ', biterr(bits, bluetoothIdealReceiver(awgn(wbr, snrVec(i)), 'BR', sps)));
end
fprintf('\n');

w2 = bluetoothWaveformGenerator(bits, 'EDR2M', sps);
fprintf('EDR2M (pi/4-DQPSK): ');
for i = 1:nSnr
    fprintf('%.4f ', biterr(bits, bluetoothIdealReceiver(awgn(w2, snrVec(i)), 'EDR2M', sps)));
end
fprintf('\n');

w3 = bluetoothWaveformGenerator(bits, 'EDR3M', sps);
fprintf('EDR3M (8-DPSK)   : ');
for i = 1:nSnr
    fprintf('%.4f ', biterr(bits, bluetoothIdealReceiver(awgn(w3, snrVec(i)), 'EDR3M', sps)));
end
fprintf('\n');
