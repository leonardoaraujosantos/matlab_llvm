% Tier-3 (Signal Processing Toolbox roadmap §4.3 tail): pulse-statistics
% follow-on — statelevels, slewrate, pulseperiod, pulsewidth, overshoot,
% undershoot, settlingtime. State levels are histogram-based; the rest
% sit on top of the existing midcross / mean-transit scaffolding.

% Three-cycle pulse train, period 20, low=-0.04, high=1.05, smoothed
% transitions at samples 11/31/51 so risetime is finite.
hi = 1.05; lo = -0.04;
x = [lo lo lo lo lo lo lo lo lo lo  0.3  hi hi hi hi hi hi hi hi hi  ...
     lo lo lo lo lo lo lo lo lo lo  0.3  hi hi hi hi hi hi hi hi hi  ...
     lo lo lo lo lo lo lo lo lo lo  0.3  hi hi hi hi hi hi hi hi lo];

% statelevels: histogram-based. Returns [low; high] 2x1.
sl = statelevels(x);
disp(sl(1));    % ~ -0.0346
disp(sl(2));    % ~  1.0446

% slewrate: rising slope = 0.8 * (hi - lo) / risetime, in signal-units / sample.
disp(slewrate(x));

% pulseperiod: distance between consecutive rising midcrosses. Should be 20.
disp(pulseperiod(x));

% pulsewidth: rising → next falling midcross.
disp(pulsewidth(x));

% overshoot / undershoot: percent above/below state levels.
disp(overshoot(x));
disp(undershoot(x));

% settlingtime: samples until x stays within d of high level after
% the rising midcross. d = 0.05 means ±5 % of (hi - lo).
disp(settlingtime(x, 0.05));
