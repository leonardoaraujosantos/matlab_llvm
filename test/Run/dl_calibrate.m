% Deep Learning T6.5 gating test — dlqcalibrate + dlqclip.  Demonstrates
% the calibration-driven workflow: collect max-abs activation across a
% calibration batch (per-layer or per-tensor), divide by 127 to get the
% int8 scale, then `dlqclip` clamps test-time activations onto that
% int8 grid.

% A handful of "calibration" samples produced by an upstream layer.
A1 = [ 1.2 -0.5  0.3];
A2 = [-1.7  0.1  0.8];
A3 = [ 0.6  1.5  2.4];

% Threading the running max: dlqcalibrate(X, runningMaxAbs) returns the
% updated max.  The convention matches the (X, scaleSoFar) tuple flow.
mx = dlqcalibrate(A1, 0);
mx = dlqcalibrate(A2, mx);
mx = dlqcalibrate(A3, mx);
mx_v = mx(1);
% Highest |value| in {1.2, -0.5, 0.3, -1.7, 0.1, 0.8, 0.6, 1.5, 2.4} = 2.4.
fprintf('calibrated max-abs x10 rounds to %.0f\n', round(10 * mx_v));

% Convert to the int8 scale.
scale = mx_v / 127.0;
fprintf('scale x10000 rounds to %.0f\n', round(10000 * scale));

% Clip a test activation onto the grid.  Should round x = 1.0 to the
% nearest multiple of scale, with magnitude no greater than 127*scale = mx.
x_in   = [ 1.0 -2.0  3.0  0.05];   % 3.0 > mx, must clip to mx.
x_out  = dlqclip(x_in, scale);

% Check: clipped value at +3.0 should be exactly 2.4 (the calibrated max).
fprintf('clip(3.0) x10 rounds to %.0f\n', round(10 * x_out(3)));
% Each value should be an integer multiple of `scale` (round-to-nearest
% lattice).
worst = 0;
for k = 1:4
    q = round(x_out(k) / scale);
    lattice_err = abs(x_out(k) - q * scale);
    if lattice_err > worst; worst = lattice_err; end
end
on_grid = 0;
if worst < 1e-9
    on_grid = 1;
end
fprintf('clipped values on int8 grid = %.0f\n', on_grid);
