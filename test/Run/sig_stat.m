% Tier-3 (Signal Processing Toolbox roadmap §4.3): pulse statistics —
% midcross, risetime, falltime, dutycycle. Default 10%/50%/90% reference
% levels, state levels auto-detected as min/max of x.

% Square wave-like signal: low for 4 samples, high for 4, low for 4, high for 4.
% Smooth ramps in/out so risetime/falltime are well-defined.
%   index 1..4: 0   (low)
%   index 5,6:  ramp up to 1
%   index 7..10: 1  (high)
%   index 11,12: ramp down to 0
%   index 13..16: 0 (low)
%   index 17,18: ramp up to 1
%   index 19..22: 1 (high)
x = [0 0 0 0 0.25 0.75 1 1 1 1 0.75 0.25 0 0 0 0 0.25 0.75 1 1 1 1];

% midcross — find sub-sample crossings of the 50% level. Should be 3
% crossings: rising near sample 5.5, falling near sample 11.5, rising
% near sample 17.5.
mc = midcross(x);
disp(size(mc, 1));      % 3
disp(mc);

% risetime: 10%→90%, averaged across the 2 rising transitions.
disp(risetime(x));

% falltime: 90%→10%, single falling transition.
disp(falltime(x));

% dutycycle: fraction of period at high state.
disp(dutycycle(x));
