% Sensor Fusion Tier-6 — track fusion + tracking-quality metrics.
% Two virtual sensors observe 3 targets; we fuse their state estimates with
% trackFuser (covariance intersection), then score each sensor and the
% fusion against ground truth using GOSPA / OSPA.
%
% Closes the Tier-6 headline `track_fusion_metrics.m` per
% docs/sensor_fusion_toolbox_roadmap.md.  The classdef `trackFuser` carrier
% over a *vector* of source tracks (full track-to-track fusion API) is a
% documented follow-on; this demo exercises the per-target CI primitive
% directly.

% Ground truth — 3 targets at staggered positions.
truth = [10.0,  0.0;
         20.0,  5.0;
         30.0, -5.0];

% Sensor A — modest noise (radar-like).
sensorA = [10.5,  0.3;
           19.7,  4.6;
           30.6, -5.4];

% Sensor B — slightly larger noise (lidar-like for the radar-only variant).
sensorB = [ 9.6, -0.4;
           20.4,  5.5;
           29.5, -4.6];

% --- Per-sensor scoring ---------------------------------------------------
cutoff = 5.0;
p      = 2.0;
g_a = trackGOSPAMetric(sensorA, truth, cutoff, p);
g_b = trackGOSPAMetric(sensorB, truth, cutoff, p);
o_a = trackOSPAMetric(sensorA, truth, cutoff, p);
o_b = trackOSPAMetric(sensorB, truth, cutoff, p);
fprintf('Per-sensor metrics (lower is better):\n');
fprintf('  GOSPA(A) = %.3f   OSPA(A) = %.3f\n', g_a(1), o_a(1));
fprintf('  GOSPA(B) = %.3f   OSPA(B) = %.3f\n', g_b(1), o_b(1));

% --- Track-to-track fusion via covariance intersection -------------------
% Per-target uncertainty (the two sensors' P diagonals).
fprintf('\nCovariance-intersection fusion (per target):\n');
fused = zeros(3, 2);
for i = 1:3
    xA = [sensorA(i, 1); sensorA(i, 2)];
    xB = [sensorB(i, 1); sensorB(i, 2)];
    % Diagonal covariances - sensor A tighter, sensor B looser.
    PA = [0.4, 0.0; 0.0, 0.4];
    PB = [0.8, 0.0; 0.0, 0.8];
    F = trackFuser(xA, PA, xB, PB);
    fused(i, 1) = F(1);
    fused(i, 2) = F(2);
    fprintf('  target %d fused = [%.3f, %.3f]\n', i, F(1), F(2));
end

% --- Fused-vs-truth GOSPA -----------------------------------------------
g_f = trackGOSPAMetric(fused, truth, cutoff, p);
fprintf('\nGOSPA(fused vs truth) = %.3f\n', g_f(1));
fprintf('Improvement vs sensor A: %.3f\n', g_a(1) - g_f(1));
