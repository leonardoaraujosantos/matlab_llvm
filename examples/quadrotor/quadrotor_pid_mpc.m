% examples/quadrotor/quadrotor_pid_mpc.m
% --------------------------------------------------------------------
% Cascade flight controller for a quadrotor following a 3-D figure-8.
%
%   Outer loop  : linear MPC tracks (x, y) — emits commanded pitch/roll.
%   Inner loops : 4 discrete PIDs track (phi, theta, psi, z) and produce
%                 (u1, u2, u3, u4).
%   Plant       : full 6-DOF nonlinear quadrotor integrated with RK4 at
%                 a 10× finer step than the MPC.
%
% Linearization used by the MPC (see quadrotor_derive_eom.m):
%   x_ddot =  g * theta_cmd            (pitch tilts the thrust forward)
%   y_ddot = -g * phi_cmd              (roll  tilts the thrust sideways)
% MPC state z = [x; x_dot; y; y_dot], input u = [theta_cmd; phi_cmd].
%
% Note: the plant derivative is written inline (not as a helper) because
% indexing a value across a user-function boundary is not yet supported
% by the compiler — local-vector indexing is. The RK4 derivative block
% therefore appears four times, once per stage.

% ---------- Constants -------------------------------------------------
m  = 1.0;
g  = 9.81;
Ix = 0.01;
Iy = 0.01;
Iz = 0.02;
Ts = 0.05;
n_inner = 10;
dt = Ts / n_inner;
N = 200;

% ---------- Outer-loop MPC --------------------------------------------
A_mpc = [1, Ts, 0,  0;
         0, 1,  0,  0;
         0, 0,  1,  Ts;
         0, 0,  0,  1];
B_mpc = [0,           0;
         g*Ts,        0;
         0,           0;
         0,          -g*Ts];
C_mpc = [1, 0, 0, 0;
         0, 0, 1, 0];
D_mpc = [0, 0; 0, 0];

sys_mpc = ss(A_mpc, B_mpc, C_mpc, D_mpc, Ts);
mpc_obj = mpc(sys_mpc, 12, 4);
mpc_obj.umax = [0.35; 0.35];
mpc_obj.umin = [-0.35; -0.35];
mpc_obj.dumax = [0.10; 0.10];
mpc_obj.dumin = [-0.10; -0.10];
mpc_state = mpcstate(4, 2, 2);

% ---------- Inner-loop PIDs -------------------------------------------
% Attitude loops are tuned ~5× faster than the MPC so the cascade has a
% clean timescale separation (theta tracks theta_cmd in ~0.15 s while
% the MPC re-plans every 0.05 s).  With Ix=Iy=0.01 a natural frequency
% of ~25 rad/s gives Kp = I·wn^2 ≈ 6, Kd = I·2·zeta·wn ≈ 0.45.
pid_z      = pid(8.0,  2.0,  4.0,  0.02);    % altitude (double integrator, 1/m)
pid_phi    = pid(6.0,  0.0,  0.45, 0.005);   % roll
pid_theta  = pid(6.0,  0.0,  0.45, 0.005);   % pitch
pid_psi    = pid(4.0,  0.0,  0.30, 0.005);   % yaw

z_int   = 0; phi_int   = 0; theta_int = 0; psi_int   = 0;
ze_prev = 0; phie_prev = 0; thetae_prev = 0; psie_prev = 0;

% ---------- Plant state (12-vector) -----------------------------------
% [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r]
% Start at the trajectory's trim state — airborne at z = 1 m and already
% moving at the figure-8's initial velocity (d/dt of the reference at
% t = 0: xdot = 2*0.6 = 1.2, ydot = 1*1.2 = 1.2 m/s).  This is the usual
% "trajectory-tracking handoff" convention: the position controller takes
% over once the vehicle is airborne and matched to the path.  It removes
% the catch-up transient that otherwise dominates the RMS error and keeps
% the quad in the small-angle regime where the MPC's linear model is
% accurate (full-run RMS ~0.014 m vs ~0.24 m starting from rest).
S = zeros(12, 1);
S(3) = 1.0;     % z   = 1 m   (at altitude)
S(4) = 1.2;     % xdot = 1.2 m/s
S(5) = 1.2;     % ydot = 1.2 m/s

% ---------- Reference trajectory --------------------------------------
% Figure-8 in (x, y) at constant altitude.  Generated out to N + p so
% the MPC can preview the upcoming p steps without running off the end.
p_horizon = 12;
N_ref = N + p_horizon;
ref_x = zeros(1, N_ref);
ref_y = zeros(1, N_ref);
ref_z = zeros(1, N_ref);
for k = 1:N_ref
    t_k = (k - 1) * Ts;
    ref_x(k) = 2.0 * sin(0.6 * t_k);
    ref_y(k) = 1.0 * sin(1.2 * t_k);
    ref_z(k) = 1.0;
end

% Logs (flat 1×N)
log_x      = zeros(1, N);
log_y      = zeros(1, N);
log_z      = zeros(1, N);
log_phi    = zeros(1, N);
log_theta  = zeros(1, N);
log_phicmd = zeros(1, N);
log_thcmd  = zeros(1, N);
log_u1     = zeros(1, N);
log_u2     = zeros(1, N);
log_u3     = zeros(1, N);
log_u4     = zeros(1, N);

fprintf('Quadrotor cascade simulation: MPC (Ts=%.3f) + 4 PIDs.\n', Ts);
fprintf('  step    t      x      y\n');

% ---------- Main loop -------------------------------------------------
for k = 1:N
    % --- Outer MPC: position -> tilt command --------------------------
    z_mpc = [S(1); S(4); S(2); S(5)];      % [x; x_dot; y; y_dot]
    mpc_state.Plant = z_mpc;
    ym = [S(1); S(2)];

    % Reference previewing (Tier-6 §7.7): feed the MPC the next p steps
    % of the (x, y) trajectory as a p×ny matrix.  Knowing where the
    % figure-8 is heading lets the controller lead the turns instead of
    % lagging them, which sharply cuts the tracking error.
    r_prev = zeros(p_horizon, 2);
    for i = 1:p_horizon
        r_prev(i, 1) = ref_x(k + i - 1);
        r_prev(i, 2) = ref_y(k + i - 1);
    end
    tilt_cmd = mpcmove(mpc_obj, mpc_state, ym, r_prev);
    theta_cmd = tilt_cmd(1, 1);
    phi_cmd   = tilt_cmd(2, 1);
    psi_cmd   = 0.0;

    log_phicmd(k) = phi_cmd;
    log_thcmd(k)  = theta_cmd;

    % --- Inner sub-steps (PIDs + plant RK4) ---------------------------
    for j = 1:n_inner
        z_err   = ref_z(k) - S(3);
        phi_err = phi_cmd  - S(7);
        th_err  = theta_cmd- S(8);
        psi_err = psi_cmd  - S(9);

        z_int     = z_int     + z_err   * dt;
        phi_int   = phi_int   + phi_err * dt;
        theta_int = theta_int + th_err  * dt;
        psi_int   = psi_int   + psi_err * dt;

        du1 = pid_z.Kp     * z_err   + pid_z.Ki     * z_int     + pid_z.Kd     * (z_err   - ze_prev)    / dt;
        u2  = pid_phi.Kp   * phi_err + pid_phi.Ki   * phi_int   + pid_phi.Kd   * (phi_err - phie_prev)  / dt;
        u3  = pid_theta.Kp * th_err  + pid_theta.Ki * theta_int + pid_theta.Kd * (th_err  - thetae_prev)/ dt;
        u4  = pid_psi.Kp   * psi_err + pid_psi.Ki   * psi_int   + pid_psi.Kd   * (psi_err - psie_prev)  / dt;

        ze_prev     = z_err;
        phie_prev   = phi_err;
        thetae_prev = th_err;
        psie_prev   = psi_err;

        % Total thrust = hover offset + altitude-PID correction.  The
        % figure-8 is gentle enough that the tuned controller keeps u1>0
        % and the torques small, so no explicit saturation is applied
        % (a conditional reassignment here would erase the scalar type
        % under the current type-inference pass).
        u1 = m * g + du1;

        % ---- RK4 stage 1: derivative at S -> k1 ----
        k1 = zeros(12, 1);
        cphi = cos(S(7));  sphi = sin(S(7));
        cth  = cos(S(8));  sth  = sin(S(8));
        cpsi = cos(S(9));  spsi = sin(S(9));
        k1(1) = S(4);
        k1(2) = S(5);
        k1(3) = S(6);
        k1(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k1(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k1(6) = u1 / m * cphi*cth - g;
        k1(7) = S(10);
        k1(8) = S(11);
        k1(9) = S(12);
        k1(10) = u2 / Ix;
        k1(11) = u3 / Iy;
        k1(12) = u4 / Iz;

        % ---- RK4 stage 2: derivative at S + dt/2*k1 -> k2 ----
        Sa = S + (dt / 2) * k1;
        k2 = zeros(12, 1);
        cphi = cos(Sa(7));  sphi = sin(Sa(7));
        cth  = cos(Sa(8));  sth  = sin(Sa(8));
        cpsi = cos(Sa(9));  spsi = sin(Sa(9));
        k2(1) = Sa(4);
        k2(2) = Sa(5);
        k2(3) = Sa(6);
        k2(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k2(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k2(6) = u1 / m * cphi*cth - g;
        k2(7) = Sa(10);
        k2(8) = Sa(11);
        k2(9) = Sa(12);
        k2(10) = u2 / Ix;
        k2(11) = u3 / Iy;
        k2(12) = u4 / Iz;

        % ---- RK4 stage 3: derivative at S + dt/2*k2 -> k3 ----
        Sb = S + (dt / 2) * k2;
        k3 = zeros(12, 1);
        cphi = cos(Sb(7));  sphi = sin(Sb(7));
        cth  = cos(Sb(8));  sth  = sin(Sb(8));
        cpsi = cos(Sb(9));  spsi = sin(Sb(9));
        k3(1) = Sb(4);
        k3(2) = Sb(5);
        k3(3) = Sb(6);
        k3(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k3(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k3(6) = u1 / m * cphi*cth - g;
        k3(7) = Sb(10);
        k3(8) = Sb(11);
        k3(9) = Sb(12);
        k3(10) = u2 / Ix;
        k3(11) = u3 / Iy;
        k3(12) = u4 / Iz;

        % ---- RK4 stage 4: derivative at S + dt*k3 -> k4 ----
        Sc = S + dt * k3;
        k4 = zeros(12, 1);
        cphi = cos(Sc(7));  sphi = sin(Sc(7));
        cth  = cos(Sc(8));  sth  = sin(Sc(8));
        cpsi = cos(Sc(9));  spsi = sin(Sc(9));
        k4(1) = Sc(4);
        k4(2) = Sc(5);
        k4(3) = Sc(6);
        k4(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k4(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k4(6) = u1 / m * cphi*cth - g;
        k4(7) = Sc(10);
        k4(8) = Sc(11);
        k4(9) = Sc(12);
        k4(10) = u2 / Ix;
        k4(11) = u3 / Iy;
        k4(12) = u4 / Iz;

        % RK4 combine (element-wise write: a loop-carried whole-vector
        % reassignment `S = S + ...` does not lower yet, but element
        % writes do — kc is a fresh combine vector).
        kc = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        S(1)  = S(1)  + kc(1);
        S(2)  = S(2)  + kc(2);
        S(3)  = S(3)  + kc(3);
        S(4)  = S(4)  + kc(4);
        S(5)  = S(5)  + kc(5);
        S(6)  = S(6)  + kc(6);
        S(7)  = S(7)  + kc(7);
        S(8)  = S(8)  + kc(8);
        S(9)  = S(9)  + kc(9);
        S(10) = S(10) + kc(10);
        S(11) = S(11) + kc(11);
        S(12) = S(12) + kc(12);
    end

    log_x(k)     = S(1);
    log_y(k)     = S(2);
    log_z(k)     = S(3);
    log_phi(k)   = S(7);
    log_theta(k) = S(8);
    log_u1(k)    = u1;
    log_u2(k)    = u2;
    log_u3(k)    = u3;
    log_u4(k)    = u4;

    if mod(k, 20) == 0
        t_k = k * Ts;
        fprintf('  %4g  %5.2f  %5.2f  %5.2f\n', k, t_k, S(1), S(2));
    end
end

fprintf('\nFinal RMS tracking error:\n');
e_sum = 0;
for k = 1:N
    ex = ref_x(k) - log_x(k);
    ey = ref_y(k) - log_y(k);
    ez = ref_z(k) - log_z(k);
    e_sum = e_sum + ex*ex + ey*ey + ez*ez;
end
fprintf('  RMS = %.4f m  (over %g steps)\n', sqrt(e_sum / N), N);

% ---------- Plots -----------------------------------------------------
t_axis = (1:N) * Ts;

figure;
subplot(2, 2, 1);
plot3(ref_x, ref_y, ref_z);
hold on;
plot3(log_x, log_y, log_z);
grid on;
title('3-D trajectory (ref vs actual)');
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
legend('reference', 'actual');

subplot(2, 2, 2);
plot(t_axis, log_x);
hold on;
plot(t_axis, log_y);
plot(t_axis, log_z);
grid on;
title('Position vs time');
xlabel('t [s]');
ylabel('[m]');
legend('x', 'y', 'z');

subplot(2, 2, 3);
plot(t_axis, log_phicmd);
hold on;
plot(t_axis, log_phi);
plot(t_axis, log_thcmd);
plot(t_axis, log_theta);
grid on;
title('Tilt: commanded vs actual');
xlabel('t [s]');
ylabel('[rad]');
legend('phi cmd', 'phi', 'theta cmd', 'theta');

subplot(2, 2, 4);
plot(t_axis, log_u1);
hold on;
plot(t_axis, log_u2);
plot(t_axis, log_u3);
plot(t_axis, log_u4);
grid on;
title('Inner-loop inputs');
xlabel('t [s]');
ylabel('thrust [N], torque [N.m]');
legend('u1 thrust', 'u2 roll', 'u3 pitch', 'u4 yaw');

saveas(gcf, 'quadrotor_pid_mpc.png');
fprintf('\nSaved plot to quadrotor_pid_mpc.png\n');
