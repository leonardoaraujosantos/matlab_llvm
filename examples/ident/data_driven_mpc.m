% data_driven_mpc.m — System Identification Toolbox Tier-3 headline.
%
% The canonical *data-driven control* workflow (User's Guide Ch.21
% "Using Identified Models for Control Design"): identify a plant from
% measured input/output data, validate it, then hand the identified
% model straight to a model-based controller — coupling System ID to
% TWO already-shipped toolboxes (Control System + Model Predictive
% Control):
%
%   iddata -> ssest (subspace) -> compare -> ss(idsys) -> mpc -> sim
%
% No first-principles model is ever written: the controller is designed
% entirely from data.

% ----- 1.  Collect input/output data from the (unknown) plant ----------
% True discrete plant (stand-in for a measured rig), sampled at 0.1 s:
%   y(t) = 1.5 y(t-1) - 0.7 y(t-2) + 1.0 u(t-1) + 0.5 u(t-2)
N  = 600;
u  = zeros(N, 1);
sd = 271828;
for k = 1:N
    sd   = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd / 2147483648 - 0.5);          % PRBS excitation
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 1.5 * y(k-1) - 0.7 * y(k-2) + 1.0 * u(k-1) + 0.5 * u(k-2);
end
Ts = 0.1;
z  = iddata(y, u, Ts);
fprintf('Collected %.0f I/O samples at Ts = %.1f s\n', N, Ts);

% ----- 2.  Identify a state-space model from the data ------------------
sys = ssest(z, 2);                 % subspace estimate, order 2
fprintf('Identified order nx = %.0f, Ts = %.1f s\n', size(sys.A, 1), sys.Ts);

% ----- 3.  Validate the fit -------------------------------------------
fit = compare(z, sys);
fprintf('Validation fit (NRMSE) = %.1f %%\n', fit);

% ----- 4.  Hand the identified model to the MPC designer ---------------
P    = ss(sys);                    % idss -> CST ss (carries discrete Ts)
ctrl = mpc(P, 10, 3);              % prediction horizon 10, control horizon 3

% ----- 5.  Closed-loop step to a unit setpoint ------------------------
r  = 1.0;
yc = sim(ctrl, 30, r);
fprintf('\nData-driven MPC closed-loop step (setpoint r = 1.0):\n');
fprintf('  t = 0.5 s : y = %.3f\n', yc(5, 1));
fprintf('  t = 1.0 s : y = %.3f\n', yc(10, 1));
fprintf('  t = 2.0 s : y = %.3f\n', yc(20, 1));
fprintf('  t = 3.0 s : y = %.3f\n', yc(30, 1));
