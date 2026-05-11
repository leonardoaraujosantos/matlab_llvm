% CommCRCDetector smoke test — wraps `crcCheck` + `crcStrip`.
%
% Builds a CRC-8-protected codeword, detects on a clean copy + a
% corrupted copy, verifies `ErrorCount` accumulates and `reset(det)`
% clears it.

msg = [1; 0; 1; 1];
codeword = crcGenerate(msg, 7, 8);
disp(length(codeword));         % 12

det = CommCRCDetector(7, 8);
recovered = det(codeword);
disp(length(recovered));        % 4
disp(det.ErrorCount);            % 0 (clean codeword)

% Corrupt one bit; the detector should flag it.
codeword(2) = 1 - codeword(2);
recovered = det(codeword);
disp(det.ErrorCount);            % 1 (error detected)

reset(det);
disp(det.ErrorCount);            % 0 after reset
