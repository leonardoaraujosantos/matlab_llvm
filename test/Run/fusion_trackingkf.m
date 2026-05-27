% Sensor Fusion Tier-2 — trackingKF linear Kalman over a constvel state.
% Recover a 1-D position+velocity from noisy position measurements.
F = [1, 0.1; 0, 1];
H = [1, 0];
Q = [1e-4, 0; 0, 1e-4];
R = [0.04];
x0 = [0.0; 0.0];
kf = trackingKF(F, H, Q, R, x0);
% Truth: constant-velocity at v=1 m/s starting from p=0.
sd = 7;
pt = 0.0;
vt = 1.0;
for k = 1:50
    pt = pt + 0.1 * vt;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    nz = (sd/2147483648 - 0.5) * 0.4;
    y = [pt + nz];
    predict(kf);
    correct(kf, y);
end
xe = kf.State;
fprintf('KF pos err = %.2f\n', xe(1) - pt);
fprintf('KF vel err = %.2f\n', xe(2) - vt);
