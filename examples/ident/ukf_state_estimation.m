% ukf_state_estimation.m — System Identification Toolbox Tier-5 headline.
%
% Nonlinear state estimation (User's Guide Ch.18 "Online Estimation":
% *Nonlinear State Estimation Using Unscented Kalman Filter*).  A
% pendulum's full state [angle; rate] is reconstructed from noisy
% angle-only measurements, even though the filter starts from a wrong
% initial guess and never sees the velocity directly.
%
%   extendedKalmanFilter / unscentedKalmanFilter
%     -> predict(obj, @StateFcn) -> correct(obj, @MeasFcn, y)
%
% The StateFcn / MeasFcn are single-argument handles (the nlmpc/greyest
% ABI); the filter object carries its mutable State + StateCovariance.
% This is the project's first dynamic Kalman filtering loop (the shipped
% Control System Toolbox kalman is steady-state-gain only).

% ----- 1.  True nonlinear plant + noisy measurements ------------------
% Discrete pendulum:  x1 += 0.1 x2;  x2 -= 0.1 sin(x1).  Measure y = x1.
N  = 80;
xt = [1.2; 0.0];
ym = zeros(N, 1);
truth1 = zeros(N, 1); truth2 = zeros(N, 1);
sd = 2718;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648);
    noise = (sd/2147483648 - 0.5) * 0.05;
    truth1(k) = xt(1); truth2(k) = xt(2);
    ym(k) = xt(1) + noise;                 % angle measurement only
    xt = [xt(1) + 0.1*xt(2); xt(2) - 0.1*sin(xt(1))];
end

% ----- 2.  Filter handles ---------------------------------------------
StateFcn = @(x) [x(1) + 0.1*x(2); x(2) - 0.1*sin(x(1))];
MeasFcn  = @(x) [x(1)];

% ----- 3.  Run the UKF from a deliberately wrong initial guess --------
ukf = unscentedKalmanFilter([0.0; 0.0], eye(2), 0.0001*eye(2), [0.01]);
xu = [0; 0];
for k = 1:N
    predict(ukf, StateFcn);
    xu = correct(ukf, MeasFcn, ym(k));
end
fprintf('UKF estimate after %.0f steps:\n', N);
fprintf('  angle = %.3f   (true %.3f)\n', xu(1), truth1(N));
fprintf('  rate  = %.3f   (true %.3f)  <- never measured directly\n', ...
        xu(2), truth2(N));

% ----- 4.  Cross-check with the EKF -----------------------------------
ekf = extendedKalmanFilter([0.0; 0.0], eye(2), 0.0001*eye(2), [0.01]);
xe = [0; 0];
for k = 1:N
    predict(ekf, StateFcn);
    xe = correct(ekf, MeasFcn, ym(k));
end
fprintf('EKF estimate: angle = %.3f, rate = %.3f\n', xe(1), xe(2));
