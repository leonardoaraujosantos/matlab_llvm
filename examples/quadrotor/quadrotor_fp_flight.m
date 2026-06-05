% examples/quadrotor/quadrotor_fp_flight.m
% ---------------------------------------------------------------------------
% Quadrotor first-principles flight demo: a 6-DOF nonlinear quadrotor flying a
% climbing, descending figure-8 under a three-layer cascade controller, then
% rendered as a cool tilting-body 3-D animation.
%
% Control architecture (see quadrotor_fp_derive.m for the symbolic EOM):
%
%        position ref (climbing figure-8)
%               |
%        +-------------------+  theta_cmd, phi_cmd   +-------------------+
%        | OUTER MPC (x,y)   | --------------------> | INNER PID x3      |
%        | double-integrator |                       | roll/pitch/yaw    | -> u2,u3,u4
%        +-------------------+                       +-------------------+
%               |  z ref
%               v
%        +-------------------+   du1 (thrust trim)
%        | INNER MPC (z)     | ----------------------------------------> u1 = m g + du1
%        | altitude tracker  |
%        +-------------------+
%                                          u1..u4
%                                            v
%                                 +-----------------------+
%                                 | 6-DOF nonlinear plant |
%                                 | (RK4, 10x sub-steps)  |
%                                 +-----------------------+
%
%   * OUTER loop  - linear MPC tracks horizontal position (x, y) and emits the
%                   tilt the body must hold:  x_ddot = g*theta, y_ddot = -g*phi.
%   * INNER loop  - a PID drives EACH attitude channel (roll, pitch, yaw) to the
%                   commanded tilt, AND a second MPC drives the altitude channel
%                   (z_ddot = du1/m), so the inner loop is PID + MPC together.
%   * PLANT       - the true nonlinear rigid-body model, integrated with RK4.
%
% This script runs through the JIT/REPL with plotting + FFmpeg enabled:
%   cat examples/quadrotor/quadrotor_fp_flight.m | build/matlabc -repl /dev/stdin
% Build flags:
%   cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON
%
% Outputs:
%   /tmp/quadrotor_fp_summary.png   - 4-panel static summary
%   /tmp/quadrotor_fp_flight.mp4    - tilting-body 3-D flight animation

% ---------- Vehicle + sim constants ----------------------------------------
m  = 1.0;        % mass [kg]
g  = 9.81;       % gravity [m/s^2]
Ix = 0.01;       % roll inertia [kg.m^2]
Iy = 0.01;       % pitch inertia
Iz = 0.02;       % yaw inertia
Ts = 0.05;       % outer controller step [s]
n_inner = 10;    % plant sub-steps per controller step
dt = Ts / n_inner;
N  = 240;        % controller steps (12 s)

% ---------- OUTER loop: position MPC (x, y) --------------------------------
% Linearised translational plant about hover: x_ddot = g*theta, y_ddot = -g*phi.
% State [x; xdot; y; ydot], manipulated variables [theta_cmd; phi_cmd].
A_p = [1 Ts 0  0;
       0 1  0  0;
       0 0  1  Ts;
       0 0  0  1];
B_p = [0     0;
       g*Ts  0;
       0     0;
       0    -g*Ts];
C_p = [1 0 0 0;
       0 0 1 0];
D_p = [0 0; 0 0];
sys_p = ss(A_p, B_p, C_p, D_p, Ts);
mpc_p = mpc(sys_p, 14, 5);                 % prediction horizon 14, control horizon 5
mpc_p.umax  = [0.35; 0.35];                % tilt bound +/- 0.35 rad (~20 deg)
mpc_p.umin  = [-0.35; -0.35];
mpc_p.dumax = [0.10; 0.10];                % tilt rate bound per step
mpc_p.dumin = [-0.10; -0.10];
st_p = mpcstate(4, 2, 2);

% ---------- INNER loop: altitude MPC (z) -----------------------------------
% Altitude double integrator z_ddot = du1/m (m = 1).  The MPC's manipulated
% variable is the thrust trim du1 added on top of the hover thrust m*g.
A_z = [1 Ts;
       0 1];
B_z = [0;
       Ts];
C_z = [1 0];
D_z = [0];
sys_z = ss(A_z, B_z, C_z, D_z, Ts);
mpc_z = mpc(sys_z, 14, 5);
mpc_z.umax = [8.0];                        % +/- 8 N thrust trim about hover
mpc_z.umin = [-8.0];
st_z = mpcstate(2, 1, 1);

% ---------- INNER loop: attitude PIDs (roll, pitch, yaw) -------------------
% Tuned ~5x faster than the MPC (wn ~ 25 rad/s) so the cascade has clean
% timescale separation.  PD form is enough for the rigid double integrators.
Kp_att = 6.0;  Kd_att = 0.45;              % roll & pitch
Kp_yaw = 4.0;  Kd_yaw = 0.30;             % yaw
phie_prev = 0; thetae_prev = 0; psie_prev = 0;

% ---------- Plant state ----------------------------------------------------
% S = [x y z  xd yd zd  phi theta psi  p q r].  Start airborne and matched to
% the figure-8's initial velocity (trajectory-tracking handoff) to stay in the
% small-angle regime where the linear MPC models are valid.
S = zeros(12, 1);
S(3) = 1.5;      % z   = 1.5 m
S(4) = 1.2;      % xdot
S(5) = 1.2;      % ydot

% ---------- Reference: a climbing / descending figure-8 --------------------
% A look-ahead of `look` steps is fed to each MPC as its setpoint so the
% controller leads the turns instead of lagging them.
look = 6;
Nr = N + look;
rx = zeros(1, Nr); ry = zeros(1, Nr); rz = zeros(1, Nr);
for k = 1:Nr
    tk = (k - 1) * Ts;
    rx(k) = 2.0 * sin(0.6 * tk);
    ry(k) = 1.0 * sin(1.2 * tk);
    rz(k) = 1.5 + 0.6 * sin(0.4 * tk);
end

% ---------- Logs -----------------------------------------------------------
log_x = zeros(1, N); log_y = zeros(1, N); log_z = zeros(1, N);
log_phi = zeros(1, N); log_theta = zeros(1, N); log_psi = zeros(1, N);
log_phicmd = zeros(1, N); log_thcmd = zeros(1, N);
log_u1 = zeros(1, N); log_u2 = zeros(1, N); log_u3 = zeros(1, N); log_u4 = zeros(1, N);

fprintf('Quadrotor cascade: OUTER MPC (x,y) + INNER PID (att) + INNER MPC (z).\n');
fprintf('  step    t       x       y       z\n');

% ---------- Main control loop ----------------------------------------------
e_sum = 0;
for k = 1:N
    % --- OUTER MPC: horizontal position -> tilt command ---
    st_p.Plant = [S(1); S(4); S(2); S(5)];
    y_pos = [S(1); S(2)];
    r_pos = [rx(k + look); ry(k + look)];
    tilt = mpcmove(mpc_p, st_p, y_pos, r_pos);
    theta_cmd = tilt(1, 1);
    phi_cmd   = tilt(2, 1);

    % --- INNER MPC: altitude -> thrust trim, then total thrust ---
    st_z.Plant = [S(3); S(6)];
    r_alt = [rz(k + look)];
    du1v = mpcmove(mpc_z, st_z, [S(3)], r_alt);
    du1 = du1v(1, 1);
    u1  = m * g + du1;

    % --- INNER PIDs + nonlinear plant at the fast rate ---
    for j = 1:n_inner
        phi_err = phi_cmd  - S(7);
        th_err  = theta_cmd - S(8);
        psi_err = 0.0      - S(9);
        u2 = Kp_att * phi_err + Kd_att * (phi_err - phie_prev) / dt;
        u3 = Kp_att * th_err  + Kd_att * (th_err  - thetae_prev) / dt;
        u4 = Kp_yaw * psi_err + Kd_yaw * (psi_err - psie_prev) / dt;
        phie_prev = phi_err; thetae_prev = th_err; psie_prev = psi_err;

        % RK4 on the 6-DOF model.  The derivative block is written inline per
        % stage (indexing across a user-function boundary is not yet lowered).
        k1 = zeros(12, 1);
        cphi=cos(S(7)); sphi=sin(S(7)); cth=cos(S(8)); sth=sin(S(8)); cpsi=cos(S(9)); spsi=sin(S(9));
        k1(1)=S(4); k1(2)=S(5); k1(3)=S(6);
        k1(4)=u1/m*(cphi*sth*cpsi+sphi*spsi); k1(5)=u1/m*(cphi*sth*spsi-sphi*cpsi); k1(6)=u1/m*cphi*cth-g;
        k1(7)=S(10); k1(8)=S(11); k1(9)=S(12); k1(10)=u2/Ix; k1(11)=u3/Iy; k1(12)=u4/Iz;
        Sa = S + (dt/2)*k1;
        k2 = zeros(12, 1);
        cphi=cos(Sa(7)); sphi=sin(Sa(7)); cth=cos(Sa(8)); sth=sin(Sa(8)); cpsi=cos(Sa(9)); spsi=sin(Sa(9));
        k2(1)=Sa(4); k2(2)=Sa(5); k2(3)=Sa(6);
        k2(4)=u1/m*(cphi*sth*cpsi+sphi*spsi); k2(5)=u1/m*(cphi*sth*spsi-sphi*cpsi); k2(6)=u1/m*cphi*cth-g;
        k2(7)=Sa(10); k2(8)=Sa(11); k2(9)=Sa(12); k2(10)=u2/Ix; k2(11)=u3/Iy; k2(12)=u4/Iz;
        Sb = S + (dt/2)*k2;
        k3 = zeros(12, 1);
        cphi=cos(Sb(7)); sphi=sin(Sb(7)); cth=cos(Sb(8)); sth=sin(Sb(8)); cpsi=cos(Sb(9)); spsi=sin(Sb(9));
        k3(1)=Sb(4); k3(2)=Sb(5); k3(3)=Sb(6);
        k3(4)=u1/m*(cphi*sth*cpsi+sphi*spsi); k3(5)=u1/m*(cphi*sth*spsi-sphi*cpsi); k3(6)=u1/m*cphi*cth-g;
        k3(7)=Sb(10); k3(8)=Sb(11); k3(9)=Sb(12); k3(10)=u2/Ix; k3(11)=u3/Iy; k3(12)=u4/Iz;
        Sc = S + dt*k3;
        k4 = zeros(12, 1);
        cphi=cos(Sc(7)); sphi=sin(Sc(7)); cth=cos(Sc(8)); sth=sin(Sc(8)); cpsi=cos(Sc(9)); spsi=sin(Sc(9));
        k4(1)=Sc(4); k4(2)=Sc(5); k4(3)=Sc(6);
        k4(4)=u1/m*(cphi*sth*cpsi+sphi*spsi); k4(5)=u1/m*(cphi*sth*spsi-sphi*cpsi); k4(6)=u1/m*cphi*cth-g;
        k4(7)=Sc(10); k4(8)=Sc(11); k4(9)=Sc(12); k4(10)=u2/Ix; k4(11)=u3/Iy; k4(12)=u4/Iz;
        kc = (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        S(1)=S(1)+kc(1); S(2)=S(2)+kc(2); S(3)=S(3)+kc(3); S(4)=S(4)+kc(4); S(5)=S(5)+kc(5); S(6)=S(6)+kc(6);
        S(7)=S(7)+kc(7); S(8)=S(8)+kc(8); S(9)=S(9)+kc(9); S(10)=S(10)+kc(10); S(11)=S(11)+kc(11); S(12)=S(12)+kc(12);
    end

    log_x(k)=S(1); log_y(k)=S(2); log_z(k)=S(3);
    log_phi(k)=S(7); log_theta(k)=S(8); log_psi(k)=S(9);
    log_phicmd(k)=phi_cmd; log_thcmd(k)=theta_cmd;
    log_u1(k)=u1; log_u2(k)=u2; log_u3(k)=u3; log_u4(k)=u4;

    ex = rx(k)-S(1); ey = ry(k)-S(2); ez = rz(k)-S(3);
    e_sum = e_sum + ex*ex + ey*ey + ez*ez;
    if mod(k, 40) == 0
        fprintf('  %4g  %5.2f  %6.3f  %6.3f  %6.3f\n', k, k*Ts, S(1), S(2), S(3));
    end
end
fprintf('Closed-loop RMS position error = %.4f m over %g steps.\n', sqrt(e_sum/N), N);

% ---------- Static 4-panel summary -----------------------------------------
t_axis = (1:N) * Ts;
ref_x = rx(1:N); ref_y = ry(1:N); ref_z = rz(1:N);

figure(1);
subplot(2, 2, 1);
plot3(ref_x, ref_y, ref_z, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
hold on;
plot3(log_x, log_y, log_z, 'b-', 'LineWidth', 1.5);
hold off;
grid on;
title('3-D trajectory: reference vs actual');
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('reference', 'actual');

subplot(2, 2, 2);
plot(t_axis, log_x, 'r-');
hold on;
plot(t_axis, log_y, 'g-');
plot(t_axis, log_z, 'b-');
hold off;
grid on;
title('Position vs time');
xlabel('t [s]'); ylabel('[m]');
legend('x', 'y', 'z');

subplot(2, 2, 3);
plot(t_axis, log_thcmd, 'r--');
hold on;
plot(t_axis, log_theta, 'r-');
plot(t_axis, log_phicmd, 'b--');
plot(t_axis, log_phi, 'b-');
hold off;
grid on;
title('Attitude PID tracking: cmd vs actual');
xlabel('t [s]'); ylabel('[rad]');
legend('theta cmd', 'theta', 'phi cmd', 'phi');

subplot(2, 2, 4);
plot(t_axis, log_u1, 'k-');
hold on;
plot(t_axis, log_u2, 'r-');
plot(t_axis, log_u3, 'g-');
plot(t_axis, log_u4, 'b-');
hold off;
grid on;
title('Inputs: thrust (MPC-z) + torques (PIDs)');
xlabel('t [s]'); ylabel('thrust [N] / torque [N.m]');
legend('u1 thrust', 'u2 roll', 'u3 pitch', 'u4 yaw');

saveas(gcf, '/tmp/quadrotor_fp_summary.png');
fprintf('Saved static summary to /tmp/quadrotor_fp_summary.png\n');

% ---------- Cool tilting-body 3-D animation --------------------------------
% Each frame redraws: the faint full reference figure-8, the actual flight path
% flown so far, a ground shadow, and the quadrotor body rotated by its true
% attitude (phi, theta, psi) so you can SEE the inner loop working the tilt.
% The camera azimuth slowly orbits the scene.
%
% NOTE: the VideoWriter setup is packed onto ONE line on purpose.  Splitting it
% across lines crashes the REPL today (see issue #236) because the writer handle
% does not survive across REPL submissions; one packed line keeps it in a single
% submission.  The render loop below can stay one-statement-per-line.
L = 0.28;                 % arm half-length [m]
stride = 2;               % animate every 2nd control step

% Start at k = 3 so the trail/shadow slices always have >= 2 points (plot3 needs
% vector, not scalar, arguments -- see issue #237).
v = VideoWriter('/tmp/quadrotor_fp_flight.mp4', 'MPEG-4'); v.FrameRate = 25; open(v); figure(2);
for k = 3:stride:N
    phi = log_phi(k); theta = log_theta(k); psi = log_psi(k);
    cphi = cos(phi); sphi = sin(phi);
    cth  = cos(theta); sth = sin(theta);
    cps  = cos(psi); sps = sin(psi);

    % Body->world rotation (ZYX).  We only need the first two body axes (the
    % arms lie in the body x-y plane), i.e. columns 1 and 2 of R.
    r11 = cps*cth;            r21 = sps*cth;            r31 = -sth;
    r12 = cps*sth*sphi - sps*cphi;
    r22 = sps*sth*sphi + cps*cphi;
    r32 = cth*sphi;

    cx = log_x(k); cy = log_y(k); cz = log_z(k);

    % Four rotor hubs in the world frame (+ configuration).
    fx = cx + r11*L; fy = cy + r21*L; fz = cz + r31*L;     % front (+body x)
    bx = cx - r11*L; by = cy - r21*L; bz = cz - r31*L;     % back
    rx2 = cx + r12*L; ry2 = cy + r22*L; rz2 = cz + r32*L;  % right (+body y)
    lx = cx - r12*L; ly = cy - r22*L; lz = cz - r32*L;     % left

    % Faint full reference path (ghost) + ground shadow of the flown path.
    plot3(ref_x, ref_y, ref_z, 'Color', [0.80 0.80 0.85], 'LineWidth', 1);
    hold on;
    plot3(log_x(1:k), log_y(1:k), zeros(1, k), 'Color', [0.85 0.85 0.85], 'LineWidth', 1);
    % Flown path so far.
    plot3(log_x(1:k), log_y(1:k), log_z(1:k), 'b-', 'LineWidth', 1.5);
    % The two arms (front arm red so heading is visible, the other blue).
    plot3([fx bx], [fy by], [fz bz], 'r-', 'LineWidth', 3);
    plot3([rx2 lx], [ry2 ly], [rz2 lz], 'b-', 'LineWidth', 3);
    % Rotor hubs and the front-rotor heading marker.
    plot3([fx bx rx2 lx], [fy by ry2 ly], [fz bz rz2 lz], 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k');
    plot3([fx fx], [fy fy], [fz fz], 'ro', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
    hold off;
    % axis([...]) 6-arg sets x/y/z limits in one call and resolves on BOTH the
    % REPL and AOT front doors; zlim() alone is missing from the AOT path (#238).
    axis([-2.6 2.6 -2.0 2.0 0 2.6]);
    view(-37.5 + k*0.35, 24);
    grid on;
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
    title('Quadrotor cascade flight: MPC(x,y) + PID(att) + MPC(z)');
    writeVideo(v, getframe(gcf));
end
close(v); fprintf('Saved flight animation to /tmp/quadrotor_fp_flight.mp4\n');
