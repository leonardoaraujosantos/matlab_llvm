% Tier-3 (Signal Processing Toolbox roadmap §4.3): envelope + hampel +
% medfilt1. All return same-shape output as input (column or row).

% medfilt1: 3-tap median filter on a spike. Spike gets replaced.
y = medfilt1([1 1 1 100 1 1 1], 3);
disp(y);

% hampel: 1.4826 * MAD threshold replaces outlier with local median.
% Single spike of 100 in an otherwise-zero signal — gets cleaned out.
yh = hampel([0 0 0 0 100 0 0 0 0], 2);
disp(yh);

% envelope of a small sawtooth: peaks at indices 2, 5; envelope
% interpolates linearly between them and holds the value at the
% endpoints.
ye = envelope([0 1 0 0 2 0 0]);
disp(ye);
