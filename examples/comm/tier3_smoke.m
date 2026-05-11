% tier3_smoke.m — exercise every Tier-3 (channel-coding) entry once.
%
% Numeric-tag conventions:
%   - CRC poly is given as an integer with the implicit leading 1
%     dropped (so CRC-16-CCITT 0x11021 -> 0x1021 = 4129 and nbits = 16).
%   - poly2trellis generator polynomials are decimal integers; pass
%     oct2dec() if you have them in octal form.
%   - vitdec opmode: 0 = truncated, 1 = terminated (force end-state 0)
%   - vitdec dectype: 0 = unquantised, 1 = hard-decision (hard ships;
%     soft is a Tier-4 follow-on).

% ----- §5.1 CRC -----
fprintf('=== CRC-16-CCITT (poly=0x1021) ===\n');
poly16 = 4129;     % 0x1021 (the 16 lowest bits — leading 1 implicit)
bits = [1; 0; 1; 1; 0; 1; 0; 0; 1; 1; 0; 1];
crc_out = crcGenerate(bits, poly16, 16);
fprintf('payload length  : %.0f\n', size(bits, 1));
fprintf('with-CRC length : %.0f (expect 28)\n', size(crc_out, 1));
fprintf('clean stream check  : %.0f (expect 0)\n', ...
        crcCheck(crc_out, poly16, 16));
% Bit-flip and re-check
err_stream = crc_out;
err_stream(3) = 1 - err_stream(3);
fprintf('corrupted stream check : %.0f (expect 1)\n', ...
        crcCheck(err_stream, poly16, 16));

% ----- §5.2 convolutional codes -----
fprintf('\n=== (171, 133)_8 convolutional code, K=7, rate 1/2 ===\n');
gens  = [oct2dec(171), oct2dec(133)];
t     = poly2trellis(7, gens);
fprintf('numStates       : %.0f (expect 64)\n', t.numStates);
fprintf('numOutputSymbols: %.0f (expect 4 = 2^n)\n', t.numOutputSymbols);

msg = [1; 0; 1; 1; 0; 0; 1; 0; 1; 0];
code = convenc(msg, t);
fprintf('code length     : %.0f (expect 20 = 10 x 2)\n', size(code, 1));
back = vitdec(code, t, 5, 0, 1);
fprintf('viterbi roundtrip ok? (errors=) : %.0f\n', ...
        biterrCount(msg, back));

% Inject a single bit error and verify the Viterbi corrects it.
noisy = code;
noisy(7) = 1 - noisy(7);
back_noisy = vitdec(noisy, t, 5, 0, 1);
fprintf('viterbi 1-bit error corrected? : %.0f\n', ...
        biterrCount(msg, back_noisy));

% ----- §5.3 Hamming (7, 4) -----
fprintf('\n=== Hamming(7,4), m=3 ===\n');
m = 3;
H = hammgenParity(m);
fprintf('H shape         : %.0f x %.0f\n', size(H, 1), size(H, 2));
msg = [1; 0; 1; 1];
code = hammingEncode(msg, m);
fprintf('encoded 7-bit   : '); disp(code');

% Each bit position singularly correctable. We rebuild the codeword
% fresh per iteration because matrix indexed assignment aliases the
% underlying buffer in this runtime (so `err = code; err(i) = ...`
% mutates both copies).
fprintf('decode 1-bit-flip at each position (errors per attempt):\n');
for pos = 1:7
    err_code = hammingEncode(msg, m);
    err_code(pos) = 1 - err_code(pos);
    back = hammingDecode(err_code, m);
    fprintf('  flip pos %.0f -> errors %.0f\n', pos, biterrCount(msg, back));
end

% ----- §5.5 block interleavers -----
fprintf('\n=== block interleaver ===\n');
data = [10; 20; 30; 40; 50];
perm = [3; 5; 1; 4; 2];
intd = intrlv(data, perm);
fprintf('interleaved     : '); disp(intd');
deint = deintrlv(intd, perm);
fprintf('round-trip      : '); disp(deint');
fprintf('errors          : %.0f\n', biterrCount(data, deint));
