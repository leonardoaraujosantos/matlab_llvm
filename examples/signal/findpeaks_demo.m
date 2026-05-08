% Peak detection via findpeaks + pulse statistics.
%
% Build a 3-pulse train (period 20 samples, low ≈ 0, high ≈ 1, with
% smoothed transitions) and exercise the §4.3 measurement surface:
%   findpeaks  → local maxima
%   statelevels → histogram-based low / high state estimation
%   pulseperiod / pulsewidth / risetime / falltime / dutycycle

% Three full cycles, period 20 samples (60 total).
hi = 1.0; lo = 0.0;
x = [lo lo lo lo lo lo lo lo lo lo  0.5  hi hi hi hi hi hi hi hi hi  ...
     lo lo lo lo lo lo lo lo lo lo  0.5  hi hi hi hi hi hi hi hi hi  ...
     lo lo lo lo lo lo lo lo lo lo  0.5  hi hi hi hi hi hi hi hi lo];

% Local maxima: in this simple signal each "high" region has its
% middle samples roughly equal — strict-monotonic findpeaks won't
% see them as peaks. To exercise findpeaks more clearly, slightly
% taper the high regions.
s = chirp((0:1/100:0.99), 5, 1, 5);  % 5 Hz tone, 100 samples
[pks, locs] = findpeaks(s);
fprintf('chirp peaks found:  %g\n', length(pks));
fprintf('first peak value:   %.4f\n', pks(1));
fprintf('first peak loc:     %g\n', locs(1));

% Pulse statistics on the rectangular train.
sl = statelevels(x);
fprintf('low  state level:   %.3f\n', sl(1));
fprintf('high state level:   %.3f\n', sl(2));
fprintf('pulse period:       %.3f\n', pulseperiod(x));
fprintf('pulse width:        %.3f\n', pulsewidth(x));
fprintf('rise time:          %.3f\n', risetime(x));
fprintf('fall time:          %.3f\n', falltime(x));
fprintf('duty cycle:         %.3f\n', dutycycle(x));
