% Scatter Plot and Eye Diagram with MATLAB Functions
%
% Ported from the MathWorks Getting Started example:
%   https://www.mathworks.com/help/comm/gs/scatter-plot-and-eye-diagram-with-matlab-functions.html
%
% Demonstrates the canonical "modulate → channel → inspect"
% digital-baseband flow on a QPSK constellation:
%
%   1. Generate uniformly-distributed bits in [0, M-1].
%   2. Map bits to QPSK symbols via `pskmod` (Gray-coded, π/4 offset).
%   3. Pass through an AWGN channel at a chosen SNR.
%   4. Read out the noisy constellation via `scatterplot` (which
%      returns the I/Q point matrix in this build — no GUI).
%
% Differences from the MathWorks page:
%
%   - **No pulse shaping**: the MathWorks example chains
%     `rcosdesign` + `upfirdn` between modulation and the channel.
%     Our `upfirdn` runtime entry is real-only at the moment, and
%     `modSig` is complex, so pulse shaping is skipped.  The
%     scatter plot of the un-shaped symbols still shows the QPSK
%     constellation under AWGN, which is what the page's first
%     plot demonstrates.  Once `upfirdn` gains complex-input
%     support the chain becomes the full MathWorks flow.
%
%   - **No `eyediagram`**: the eye-diagram renderer isn't wired
%     in this build.  The MathWorks page's eye-diagram plot is
%     specific to the post-pulse-shape waveform, so it doesn't
%     have a meaningful analogue for un-shaped QPSK symbols.
%
%   - **`pskmod` 4-arg form**: `pskmod(data, M, ini_phase,
%     symbolOrder)` — `symbolOrder = 0` selects Gray coding
%     (matches MathWorks default).
%
%   - **`awgn` 2-arg form**: auto-measures input signal power
%     (equivalent to the page's `'measured'` token, which the
%     plain function-form signature doesn't accept).

% --- Configuration ---
M       = 4;          % QPSK
nSym    = 1000;       % number of symbols
snr_dB  = 15.0;       % AWGN channel SNR

% --- 1. Random bits ---
data = randi(M, nSym, 1) - 1;          % uniform in [0, M-1]

% --- 2. QPSK modulation (π/4 offset, Gray code) ---
modSig = pskmod(data, M, 0.7853981633974483, 0);    % π/4 ≈ 0.7854

% --- 3. AWGN channel ---
rxSig = awgn(modSig, snr_dB);

% --- 4. Inspect ---
%   `scatterplot` returns an N×2 matrix of [I, Q] points.  Under
%   QPSK with π/4 offset, the noise-free constellation sits at the
%   four (±1/√2, ±1/√2) corners (≈ ±0.707 along each axis).  After
%   AWGN the points form Gaussian clouds around those corners.
pts = scatterplot(rxSig);
disp(size(pts, 1));         % 1000 rows
disp(size(pts, 2));         % 2 columns

% First few I/Q points — values vary with the random seed but stay
% near the unit-circle quadrants.
disp(pts(1, 1));
disp(pts(1, 2));
disp(pts(2, 1));
disp(pts(2, 2));
disp(pts(3, 1));
disp(pts(3, 2));
