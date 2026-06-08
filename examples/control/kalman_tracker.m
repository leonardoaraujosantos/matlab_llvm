% kalman — steady-state Kalman filter for a constant-velocity tracker.
%
% Tier 3.2 (control_toolbox_roadmap.md §4.2) — NOT YET SHIPPED.
% Depends on Tier 1.5 (care — for the continuous estimator) /
% Tier 1.5 (dare — for the discrete one).
%
% Tracking model — position and velocity, position-only measurement.
%   x = [p; v]
%   xdot = [0 1; 0 0] x + [0; 1] w     (w is acceleration noise)
%   y    = [1 0] x + v                 (v is measurement noise)
%   E[w*w'] = Q,  E[v*v'] = R.
%
% The Kalman filter steady-state solves the dual Riccati to LQR:
%   A*P + P*A' - P*C'*R^{-1}*C*P + B*Q*B' = 0  (continuous CARE on P)
%   L = P * C' / R                              (Kalman gain)

A = [0 1; 0 0];
B = [0; 1];          % process-noise input
C = [1 0];           % measurement is position only
D = 0;
sys = ss(A, B, C, D);

% Process and measurement noise covariances.
Qn = 0.01;           % acceleration variance (m^2/s^4)
Rn = 0.1;            % measurement noise variance (m^2)

% --- 1. Construct the steady-state Kalman estimator.
%   kalman returns:
%     kest — estimator state-space model:
%              xhat_dot = (A - L*C) xhat + [B L] [u; y]
%              [yhat; xhat] = ...
%     L    — Kalman gain.
%     P    — steady-state state-error covariance.
[kest, L, P] = kalman(sys, Qn, Rn);
disp('Kalman gain L:');
disp(L);
disp('steady-state covariance P:');
disp(P);

% --- 2. Innovation gain — for a constant-velocity tracker the
% closed-form steady-state Kalman gain has L = [sqrt(2*r); r] where
% r = sqrt(Qn / Rn). Verify.
r_pred = sqrt(Qn / Rn);
disp('predicted L(1) ≈ sqrt(2*r):');
disp(sqrt(2 * r_pred));     % theoretical
disp('predicted L(2) ≈ r:');
disp(r_pred);

% --- 3. Discrete-time variant — for a digital tracker at Ts = 0.1 s.
Ts = 0.1;
sys_d = c2d(sys, Ts, 'zoh');
[Ad, Bd, Cd, Dd] = ssdata(sys_d);
[kest_d, Ld, Pd] = kalman(sys_d, Qn, Rn);
disp('discrete Kalman gain Ld:');
disp(Ld);
% Eigenvalues of (Ad - Ld*Cd) should lie inside the unit circle.
e = eig(Ad - Ld * Cd);
disp('discrete-estimator pole magnitudes (must be < 1):');
disp(abs(e));

% ----- simulate + plot a constant-velocity tracking run --------------
% Steady-state fixed-gain CV tracker (the alpha-beta form of the filter).
% Deterministic measurement perturbation (no RNG) so AOT and JIT match.
Nt = 60; truep = zeros(Nt,1); measp = zeros(Nt,1); estp = zeros(Nt,1);
xt = [0; 1]; ph = 0; vh = 1; alpha = 0.5; beta = 0.1;
for kk = 1:Nt
    xt = Ad * xt;                                   % true CV motion
    mvec = Cd * xt;
    meas = mvec(1,1) + 0.4*sin(0.9*kk) + 0.3*cos(1.7*kk);  % noisy position
    ph = ph + Ts * vh;                              % predict position
    resid = meas - ph;                              % innovation
    ph = ph + alpha * resid;                        % position correction
    vh = vh + (beta / Ts) * resid;                  % velocity correction
    truep(kk) = xt(1,1); measp(kk) = meas; estp(kk) = ph;
end
kidx = (1:Nt)';
figure; plot(kidx, truep, 'k-'); hold on; scatter(kidx, measp); plot(kidx, estp, 'b-'); grid on;
xlabel('step'); ylabel('position'); title('Kalman tracker: true / measured / estimate');
legend('true', 'measured', 'estimate');
saveas(gcf, '/tmp/ctrl_kalman_track.png');
