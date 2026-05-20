% System Identification Tier-5 — EKF / UKF nonlinear state estimation.
% Estimate a pendulum's [angle; rate] from noisy angle measurements,
% starting from a wrong initial guess.  Discrete dynamics:
%   x1+=0.1*x2;  x2-=0.1*sin(x1).   Measure y = x1.
N = 60;
xt = [1.0; 0.0]; ym = zeros(N, 1);
t1 = zeros(N, 1); t2 = zeros(N, 1); sd = 42;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648);
    n = (sd/2147483648 - 0.5) * 0.05;
    t1(k) = xt(1); t2(k) = xt(2);
    ym(k) = xt(1) + n;
    xt = [xt(1) + 0.1*xt(2); xt(2) - 0.1*sin(xt(1))];
end
f = @(x) [x(1) + 0.1*x(2); x(2) - 0.1*sin(x(1))];
h = @(x) [x(1)];

ekf = extendedKalmanFilter([0.0; 0.0], eye(2), 0.0001*eye(2), [0.01]);
xe = [0; 0];
for k = 1:N
    predict(ekf, f);
    xe = correct(ekf, h, ym(k));
end
fprintf('EKF x1 = %.2f (true %.2f)\n', xe(1), t1(N));
fprintf('EKF x2 = %.2f (true %.2f)\n', xe(2), t2(N));

ukf = unscentedKalmanFilter([0.0; 0.0], eye(2), 0.0001*eye(2), [0.01]);
xu = [0; 0];
for k = 1:N
    predict(ukf, f);
    xu = correct(ukf, h, ym(k));
end
fprintf('UKF x1 = %.2f (true %.2f)\n', xu(1), t1(N));
fprintf('UKF x2 = %.2f (true %.2f)\n', xu(2), t2(N));
