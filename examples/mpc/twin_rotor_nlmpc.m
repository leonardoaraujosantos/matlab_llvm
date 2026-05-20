% examples/mpc/twin_rotor_nlmpc.m — MIMO Nonlinear MPC headline.
% --------------------------------------------------------------------
% 2-DOF helicopter / twin-rotor — the canonical MIMO nonlinear control
% benchmark.  A body pivots in pitch and yaw, driven by two rotors:
%
%   * main rotor (u1) mostly produces pitch torque, but also yaws the body
%   * tail rotor (u2) mostly produces yaw torque, but also pitches it
%   * gravity applies a nonlinear restoring torque -Kg*sin(pitch)
%
% This cross-coupling is exactly why a MIMO controller is needed: a pair
% of independent SISO loops would fight each other through the off-axis
% torques.  A single nonlinear MPC coordinates both rotors against the
% coupling AND the gravity nonlinearity.
%
% State  S = [pitch; yaw; pitch_rate; yaw_rate]   (nx = 4)
% Inputs u = [main_rotor; tail_rotor]             (nu = 2)
% Output y = [pitch; yaw]                          (ny = 2)
%
% Continuous-time dynamics (Jp, Jy = pitch/yaw inertias):
%   pitch'' = (0.12*u1 + 0.02*u2 - 0.02*pitch_rate - 0.4*sin(pitch)) / 0.05
%   yaw''   = (0.10*u2 + 0.03*u1 - 0.02*yaw_rate)                    / 0.04
%
% The controller is `nlmpc(4, 2, 2)`: nonlinear MPC over the true model
% via an anonymous StateFcn (packed zxu = [x; u]), solved each step with
% the toolbox's fmincon-based NLP solver and RK4 prediction rollout.
%
% Run with matlabc built with -DMATLAB_LLVM_WITH_PLOT=ON.

% ---------- Nonlinear MPC controller ----------------------------------
nlobj = nlmpc(4, 2, 2);          % nx = 4, ny = 2, nu = 2
nlobj.Ts = 0.1;                  % control sample time
nlobj.p  = 12;                   % prediction horizon
nlobj.m  = 3;                    % control horizon
nlobj.umax = [4; 4];             % rotor command bounds
nlobj.umin = [-4; -4];
nlobj.Wy  = [3; 3];              % track pitch & yaw
nlobj.Wdu = [0.1; 0.1];          % gentle move suppression

% StateFcn as an anonymous handle.  zxu = [pitch; yaw; p_rate; y_rate;
% u1; u2], returns the 4-state derivative.  Coefficients are inline
% literals (the handle cannot capture workspace variables).
state_fn = @(zxu) [zxu(3,1); zxu(4,1); (0.12*zxu(5,1) + 0.02*zxu(6,1) - 0.02*zxu(3,1) - 0.4*sin(zxu(1,1)))/0.05; (0.10*zxu(6,1) + 0.03*zxu(5,1) - 0.02*zxu(4,1))/0.04];

% ---------- Simulation setup ------------------------------------------
Ts = 0.1;
n_inner = 5;                     % plant sub-steps per control step
dt = Ts / n_inner;
N = 80;                          % 8 s flight

% Two-phase setpoint: hold (pitch=0.3, yaw=0.5) then step to
% (pitch=-0.2, yaw=0.2) at t = 4 s — exercises MIMO re-coordination.
ref_p = zeros(1, N);
ref_y = zeros(1, N);
for k = 1:N
    if k <= 40
        ref_p(k) = 0.3;
        ref_y(k) = 0.5;
    else
        ref_p(k) = -0.2;
        ref_y(k) = 0.2;
    end
end

S = zeros(4, 1);                 % start at rest, level
u_prev1 = 0;
u_prev2 = 0;

log_p   = zeros(1, N);
log_y   = zeros(1, N);
log_u1  = zeros(1, N);
log_u2  = zeros(1, N);

fprintf('Twin-rotor MIMO nonlinear MPC: nlmpc(4,2,2), Ts=%.2f.\n', Ts);
fprintf('Phase 1 (t<4): pitch->0.3 yaw->0.5;  Phase 2: pitch->-0.2 yaw->0.2\n');
fprintf('  step    t     pitch    yaw\n');

% ---------- Closed loop -----------------------------------------------
for k = 1:N
    r = [ref_p(k); ref_y(k)];
    u_prev = [u_prev1; u_prev2];

    u = nlmpcmove(nlobj, S, u_prev, r, state_fn);
    % Copy the solver return into a fresh local vector before extracting
    % the two scalars — indexing the same builtin return at multiple
    % positions inside the loop otherwise confuses type inference.
    uu = zeros(2, 1);
    uu(1) = u(1, 1);
    uu(2) = u(2, 1);
    u1 = uu(1);
    u2 = uu(2);
    u_prev1 = u1;
    u_prev2 = u2;

    % Plant integration: RK4 on the true nonlinear model.
    for j = 1:n_inner
        k1 = zeros(4, 1);
        k1(1) = S(3);
        k1(2) = S(4);
        k1(3) = (0.12*u1 + 0.02*u2 - 0.02*S(3) - 0.4*sin(S(1)))/0.05;
        k1(4) = (0.10*u2 + 0.03*u1 - 0.02*S(4))/0.04;

        Sa = S + (dt/2)*k1;
        k2 = zeros(4, 1);
        k2(1) = Sa(3);
        k2(2) = Sa(4);
        k2(3) = (0.12*u1 + 0.02*u2 - 0.02*Sa(3) - 0.4*sin(Sa(1)))/0.05;
        k2(4) = (0.10*u2 + 0.03*u1 - 0.02*Sa(4))/0.04;

        Sb = S + (dt/2)*k2;
        k3 = zeros(4, 1);
        k3(1) = Sb(3);
        k3(2) = Sb(4);
        k3(3) = (0.12*u1 + 0.02*u2 - 0.02*Sb(3) - 0.4*sin(Sb(1)))/0.05;
        k3(4) = (0.10*u2 + 0.03*u1 - 0.02*Sb(4))/0.04;

        Sc = S + dt*k3;
        k4 = zeros(4, 1);
        k4(1) = Sc(3);
        k4(2) = Sc(4);
        k4(3) = (0.12*u1 + 0.02*u2 - 0.02*Sc(3) - 0.4*sin(Sc(1)))/0.05;
        k4(4) = (0.10*u2 + 0.03*u1 - 0.02*Sc(4))/0.04;

        kc = (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        S(1) = S(1) + kc(1);
        S(2) = S(2) + kc(2);
        S(3) = S(3) + kc(3);
        S(4) = S(4) + kc(4);
    end

    log_p(k)  = S(1);
    log_y(k)  = S(2);
    log_u1(k) = u1;
    log_u2(k) = u2;

    if mod(k, 10) == 0
        t_k = k * Ts;
        fprintf('  %4g  %5.2f  %6.3f  %6.3f\n', k, t_k, S(1), S(2));
    end
end

% ---------- Tracking error --------------------------------------------
e_sum = 0;
for k = 1:N
    ep = ref_p(k) - log_p(k);
    ey = ref_y(k) - log_y(k);
    e_sum = e_sum + ep*ep + ey*ey;
end
fprintf('\nRMS tracking error: %.4f rad  (over %g steps)\n', sqrt(e_sum / N), N);

% ---------- Plots -----------------------------------------------------
t_axis = (1:N) * Ts;

figure;
subplot(2, 2, 1);
plot(t_axis, ref_p);
hold on;
plot(t_axis, log_p);
grid on;
title('Pitch tracking');
xlabel('t [s]');
ylabel('pitch [rad]');
legend('reference', 'actual');

subplot(2, 2, 2);
plot(t_axis, ref_y);
hold on;
plot(t_axis, log_y);
grid on;
title('Yaw tracking');
xlabel('t [s]');
ylabel('yaw [rad]');
legend('reference', 'actual');

subplot(2, 2, 3);
plot(t_axis, log_u1);
hold on;
plot(t_axis, log_u2);
grid on;
title('Rotor commands');
xlabel('t [s]');
ylabel('command');
legend('u1 main', 'u2 tail');

subplot(2, 2, 4);
plot(t_axis, log_p);
hold on;
plot(t_axis, log_y);
grid on;
title('Both outputs (coupling)');
xlabel('t [s]');
ylabel('[rad]');
legend('pitch', 'yaw');

saveas(gcf, 'twin_rotor_nlmpc.png');
fprintf('Saved plot to twin_rotor_nlmpc.png\n');
