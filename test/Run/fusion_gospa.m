% Sensor Fusion Tier-6 — track fusion + GOSPA metric.
% Two-sensor covariance-intersection fusion + GOSPA score of fused tracks
% against truth.  Verifies trackFuser (CI), trackGOSPAMetric, and the RTS
% smoother (free-function form).

% Synthetic 3-target ground truth (positions only, 2-D).
Y = [10.0, 0.0;
     20.0, 5.0;
     30.0, -5.0];

% Sensor A's estimates: noisy but close to truth (radar-like).
Xa = [10.5, 0.3;
      19.7, 4.6;
      30.6, -5.4];

% Sensor B's estimates: equally noisy.
Xb = [ 9.6, -0.4;
      20.4, 5.5;
      29.5, -4.6];

% GOSPA distance for each sensor against truth.
g_a = trackGOSPAMetric(Xa, Y, 5.0, 2.0);
g_b = trackGOSPAMetric(Xb, Y, 5.0, 2.0);
fprintf('GOSPA(sensor A vs truth) = %.3f\n', g_a(1));
fprintf('GOSPA(sensor B vs truth) = %.3f\n', g_b(1));

% Covariance-intersection fusion for one of the targets.
xA = [10.5; 0.3];
xB = [ 9.6;-0.4];
PA = [0.5, 0.0; 0.0, 0.5];
PB = [0.7, 0.0; 0.0, 0.7];
F = trackFuser(xA, PA, xB, PB);
% F packs x_fused on top of vec(P_fused).
fprintf('CI fused x = [%.3f, %.3f]\n', F(1), F(2));

% trackErrorMetrics RMSE accumulator.
rmse = trackErrorMetrics(Xa, Y);
fprintf('Sensor A track RMSE = %.3f\n', rmse(1));

% rtsSmoother: build a tiny 5-step constvel forward history, then smooth.
Fm = [1, 1; 0, 1];
Xh = zeros(5, 2);
Ph = zeros(5, 4);
Xh(1,1) = 0; Xh(1,2) = 1;
Xh(2,1) = 1.2; Xh(2,2) = 1.1;
Xh(3,1) = 2.0; Xh(3,2) = 0.9;
Xh(4,1) = 2.9; Xh(4,2) = 1.0;
Xh(5,1) = 4.0; Xh(5,2) = 1.0;
for i = 1:5
    Ph(i,1) = 0.2; Ph(i,4) = 0.2;
end
Xs = rtsSmoother(Fm, Xh, Ph);
fprintf('RTS smoothed x[1] = %.3f (forward was 0.0)\n', Xs(1));
fprintf('RTS smoothed x[5] = %.3f (forward was 4.0)\n', Xs(9));
