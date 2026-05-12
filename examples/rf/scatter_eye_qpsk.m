% Scatter Plot and Eye Diagram with MATLAB Functions
%
% Ported from the MathWorks Getting Started example:
%   https://www.mathworks.com/help/comm/gs/scatter-plot-and-eye-diagram-with-matlab-functions.html
%
% Demonstrates the canonical "modulate → pulse-shape → AWGN →
% inspect" digital-baseband flow on a QPSK constellation:
%
%   1. Generate uniformly-distributed bits in [0, M-1].
%   2. Map bits to QPSK symbols via `pskmod` (Gray-coded, π/4 offset).
%   3. Upsample + pulse-shape with a root-raised-cosine filter
%      (the standard square-root Nyquist pulse for matched-filter
%      reception).
%   4. Pass through an AWGN channel at a chosen Eb/N0.
%   5. Read out the noisy constellation via `scatterplot` (which
%      returns the I/Q point matrix in this build — no GUI).
%   6. Read out the eye-diagram trace matrix via `eyediagram`.
%
% Differences from the MathWorks page:
%
%   - **Headless `scatterplot` / `eyediagram`**: both return numeric
%     trace matrices in this build (no GUI).  The shapes match what
%     a `plot` call would consume, so wiring them to the Cairo
%     backend is just a follow-on `plot(pts(:,1), pts(:,2))` and
%     `plot(eye)` call.
%
%   - **`pskmod` 4-arg form**: `pskmod(data, M, ini_phase,
%     symbolOrder)` — `symbolOrder = 0` selects Gray coding
%     (matches MathWorks default).
%
%   - **`rcosdesign` 4-arg form**: `rcosdesign(beta, span, sps,
%     shape)` — `shape = 1` selects 'sqrt' (root-raised-cosine).
%
%   - **`awgn` 2-arg form**: auto-measures input signal power
%     (equivalent to the page's `'measured'` token, which the
%     plain function-form signature doesn't accept).

% --- Configuration ---
M       = 4;          % QPSK
nSym    = 1000;       % number of symbols
sps     = 4;          % samples per symbol
beta    = 0.35;       % RRC rolloff factor
span    = 10;         % filter span in symbols
EbNo_dB = 10.0;       % per-bit SNR target (dB)

% --- 1. Random bits ---
data = randi(M, nSym, 1) - 1;          % uniform in [0, M-1]

% --- 2. QPSK modulation (π/4 offset, Gray code) ---
modSig = pskmod(data, M, 0.7853981633974483, 0);    % π/4 ≈ 0.7854
disp(length(modSig));                  % 1000 (nSym)

% --- 3. Root-raised-cosine pulse shaping ---
rrcFilter = rcosdesign(beta, span, sps, 1);    % 1 = 'sqrt'
txSig = upfirdn(modSig, rrcFilter, sps, 1);
disp(length(txSig));                   % nSym*sps + (filter_taps - 1)

% --- 4. AWGN channel ---
%  snr = EbNo + 10*log10(log2(M)) - 10*log10(sps)
snr_dB = EbNo_dB + 10.0 .* log2(M) .* (1.0 ./ log2(10)) ...
         - 10.0 .* log10(sps);
rxSig = awgn(txSig, snr_dB);

% --- 5. Inspect the received constellation ---
%   `scatterplot` returns an N×2 matrix of [I, Q] points.  After
%   pulse shaping, points in between symbol centers carry partial
%   ISI from the RRC pulse (this is what an eyediagram visualizes).
%   Around a steady-state symbol center, the points sit near the
%   QPSK corners (±1/√2, ±1/√2) plus AWGN noise.
pts = scatterplot(rxSig);
disp(size(pts, 1));                    % 4040 = 1000*4 + 40
disp(size(pts, 2));                    % 2

% Sample a steady-state point (one symbol period in, well past the
% filter delay).  The exact value varies with the random seed.
mid = floor(size(pts, 1) ./ 2);
disp(pts(mid, 1));
disp(pts(mid, 2));

% --- 6. Eye diagram ---
%   `eyediagram(x, n)` returns an `n × num_traces` matrix where each
%   column is one n-sample slice of the signal — a 2-symbol slice
%   here (2*sps = 8 samples).  An open eye means the symbol-decision
%   instants (rows near the center) cluster cleanly above and below
%   zero with little spread.
eye = eyediagram(real(rxSig), 2 .* sps);
disp(size(eye, 1));                    % 2*sps = 8 rows
disp(size(eye, 2));                    % ~nSym/2 traces (post-filter-delay)

% Mid-buffer sample at the symbol-decision instant (last row of the
% trace).  In a clean eye, values cluster near ±0.707 (QPSK levels).
disp(eye(8, floor(size(eye, 2) ./ 2)));
