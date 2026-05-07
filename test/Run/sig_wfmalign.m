% Tier-3 (Signal Processing Toolbox roadmap §4.2 + §4.4): waveform
% generators + alignment helpers. All take time-vector inputs and
% return same-shape outputs.
%
% Note: scalar inputs need to be wrapped in [..] to keep them as
% matrix-typed (otherwise Sema folds [0] to a scalar f64 that doesn't
% match the matlab_<fn>(matlab_mat *, ...) signatures).

t = [0 0.25 0.5 0.75 1.0];

% chirp linear: at t=0 phase is 0, so cos(0) = 1.
y_c = chirp(t, 1, 1, 2);
disp(y_c(1));               % 1

% sinc on a column vector. sinc(0) = 1, sinc(±integer) = 0.
disp(sinc([0 1 2 3]));      % [1 ~0 ~0 ~0]

% rectpuls peak at t=0.
disp(rectpuls([0 0.6], 1));  % [1 0]

% tripuls linear ramp.
disp(tripuls([0 0.25 0.5], 1));   % [1 0.5 0]

% gauspuls at t=0: exp(0)·cos(0) = 1.
disp(gauspuls([0 1], 1, 0.5));    % [1, decayed]

% square wave: square(0, 50) = 1; square(π·1.5, 50) wraps to -1.
disp(square([0 3.5], 50));   % [1 -1]

% sawtooth: width 0.5 gives a triangle. Sample a few points.
disp(sawtooth([0 1 2 3], 0.5));

% xcov(x, x) returns a row of length 2N-1; central element is the
% sum of mean-centred squares.
x = [1 2 3 4 5];
c = xcov(x, x);
disp(size(c, 2));            % 2N-1 = 9
disp(c(5));                  % central lag = sum((x-mean(x)).^2) = 10

% finddelay(x, x) = 0.
disp(finddelay(x, x));

% finddelay(x_delayed_by_2, x) = 2. (y[n] = x[n-2] for n ≥ 2.)
disp(finddelay([0 0 1 2 3 4 5], x));

% dtw of identical sequences: 0.
disp(dtw(x, x));
