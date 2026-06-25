% examples/quadrotor/quadrotor_pid_mpc_3d.m
% --------------------------------------------------------------------
% The cascade flight controller of quadrotor_pid_mpc.m, visualised in 3-D.
%
%   Outer loop  : linear MPC tracks (x, y) — emits commanded pitch/roll.
%   Inner loops : 4 discrete PIDs track (phi, theta, psi, z).
%   Plant       : full 6-DOF nonlinear quadrotor integrated with RK4.
%   Viewer      : the sim3d command-line API records one keyframe per
%                 control step and exports a self-contained Babylon.js
%                 player — watch the quad tilt into the figure-8 turns.
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < quadrotor_pid_mpc_3d.m
%     xdg-open quadrotor_pid_mpc_3d.html

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
pid_z      = pid(8.0,  2.0,  4.0,  0.02);
pid_phi    = pid(6.0,  0.0,  0.45, 0.005);
pid_theta  = pid(6.0,  0.0,  0.45, 0.005);
pid_psi    = pid(4.0,  0.0,  0.30, 0.005);

z_int   = 0; phi_int   = 0; theta_int = 0; psi_int   = 0;
ze_prev = 0; phie_prev = 0; thetae_prev = 0; psie_prev = 0;

% ---------- Plant state (12-vector) -----------------------------------
% [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r]
S = zeros(12, 1);
S(3) = 1.0;
S(4) = 1.2;
S(5) = 1.2;

% ---------- Reference trajectory --------------------------------------
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

% ---------- 3-D scene -------------------------------------------------
% A flat box body for the airframe plus four rotor disks at the arm tips,
% parented to the body so they follow its pose. A ground plane gives the
% flight a reference floor.
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [12 12 1];
ground.Color = [0.16 0.17 0.20];
w.add(ground);

body = sim3d.Actor('body', 'box');
body.Size = [0.55 0.55 0.10];
body.Color = [0.20 0.55 0.95];
w.add(body);

% Four rotor hubs (flattened spheres) fixed to the body via local offsets;
% parenting makes them follow the body's pose through the scene graph.
arm = 0.34;
rotor1 = sim3d.Actor('rotor1', 'sphere'); rotor1.Color = [0.95 0.75 0.15]; rotor1.Scale = [0.26 0.26 0.14]; w.add(rotor1);
rotor2 = sim3d.Actor('rotor2', 'sphere'); rotor2.Color = [0.95 0.75 0.15]; rotor2.Scale = [0.26 0.26 0.14]; w.add(rotor2);
rotor3 = sim3d.Actor('rotor3', 'sphere'); rotor3.Color = [0.90 0.30 0.25]; rotor3.Scale = [0.26 0.26 0.14]; w.add(rotor3);
rotor4 = sim3d.Actor('rotor4', 'sphere'); rotor4.Color = [0.90 0.30 0.25]; rotor4.Scale = [0.26 0.26 0.14]; w.add(rotor4);
rotor1.setParent(body); rotor1.Translation = [ arm  0    0.06];
rotor2.setParent(body); rotor2.Translation = [-arm  0    0.06];
rotor3.setParent(body); rotor3.Translation = [ 0    arm  0.06];
rotor4.setParent(body); rotor4.Translation = [ 0   -arm  0.06];

w.open();

fprintf('Quadrotor cascade simulation (3-D): MPC (Ts=%.3f) + 4 PIDs.\n', Ts);

% ---------- Main loop -------------------------------------------------
for k = 1:N
    % --- Outer MPC: position -> tilt command --------------------------
    z_mpc = [S(1); S(4); S(2); S(5)];
    mpc_state.Plant = z_mpc;
    ym = [S(1); S(2)];

    r_prev = zeros(p_horizon, 2);
    for i = 1:p_horizon
        r_prev(i, 1) = ref_x(k + i - 1);
        r_prev(i, 2) = ref_y(k + i - 1);
    end
    tilt_cmd = mpcmove(mpc_obj, mpc_state, ym, r_prev);
    theta_cmd = tilt_cmd(1, 1);
    phi_cmd   = tilt_cmd(2, 1);
    psi_cmd   = 0.0;

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

        u1 = m * g + du1;

        % ---- RK4 stage 1 ----
        k1 = zeros(12, 1);
        cphi = cos(S(7));  sphi = sin(S(7));
        cth  = cos(S(8));  sth  = sin(S(8));
        cpsi = cos(S(9));  spsi = sin(S(9));
        k1(1) = S(4);  k1(2) = S(5);  k1(3) = S(6);
        k1(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k1(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k1(6) = u1 / m * cphi*cth - g;
        k1(7) = S(10); k1(8) = S(11); k1(9) = S(12);
        k1(10) = u2 / Ix; k1(11) = u3 / Iy; k1(12) = u4 / Iz;

        % ---- RK4 stage 2 ----
        Sa = S + (dt / 2) * k1;
        k2 = zeros(12, 1);
        cphi = cos(Sa(7));  sphi = sin(Sa(7));
        cth  = cos(Sa(8));  sth  = sin(Sa(8));
        cpsi = cos(Sa(9));  spsi = sin(Sa(9));
        k2(1) = Sa(4); k2(2) = Sa(5); k2(3) = Sa(6);
        k2(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k2(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k2(6) = u1 / m * cphi*cth - g;
        k2(7) = Sa(10); k2(8) = Sa(11); k2(9) = Sa(12);
        k2(10) = u2 / Ix; k2(11) = u3 / Iy; k2(12) = u4 / Iz;

        % ---- RK4 stage 3 ----
        Sb = S + (dt / 2) * k2;
        k3 = zeros(12, 1);
        cphi = cos(Sb(7));  sphi = sin(Sb(7));
        cth  = cos(Sb(8));  sth  = sin(Sb(8));
        cpsi = cos(Sb(9));  spsi = sin(Sb(9));
        k3(1) = Sb(4); k3(2) = Sb(5); k3(3) = Sb(6);
        k3(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k3(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k3(6) = u1 / m * cphi*cth - g;
        k3(7) = Sb(10); k3(8) = Sb(11); k3(9) = Sb(12);
        k3(10) = u2 / Ix; k3(11) = u3 / Iy; k3(12) = u4 / Iz;

        % ---- RK4 stage 4 ----
        Sc = S + dt * k3;
        k4 = zeros(12, 1);
        cphi = cos(Sc(7));  sphi = sin(Sc(7));
        cth  = cos(Sc(8));  sth  = sin(Sc(8));
        cpsi = cos(Sc(9));  spsi = sin(Sc(9));
        k4(1) = Sc(4); k4(2) = Sc(5); k4(3) = Sc(6);
        k4(4) = u1 / m * (cphi*sth*cpsi + sphi*spsi);
        k4(5) = u1 / m * (cphi*sth*spsi - sphi*cpsi);
        k4(6) = u1 / m * cphi*cth - g;
        k4(7) = Sc(10); k4(8) = Sc(11); k4(9) = Sc(12);
        k4(10) = u2 / Ix; k4(11) = u3 / Iy; k4(12) = u4 / Iz;

        kc = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        S(1)  = S(1)  + kc(1);   S(2)  = S(2)  + kc(2);
        S(3)  = S(3)  + kc(3);   S(4)  = S(4)  + kc(4);
        S(5)  = S(5)  + kc(5);   S(6)  = S(6)  + kc(6);
        S(7)  = S(7)  + kc(7);   S(8)  = S(8)  + kc(8);
        S(9)  = S(9)  + kc(9);   S(10) = S(10) + kc(10);
        S(11) = S(11) + kc(11);  S(12) = S(12) + kc(12);
    end

    % --- Record one animation keyframe for this control step ----------
    body.Translation = [S(1) S(2) S(3)];
    body.Rotation = [S(7) S(8) S(9)];
    w.run(Ts);

    if mod(k, 20) == 0
        fprintf('  step %4g  t=%5.2f  pos=(%5.2f, %5.2f, %5.2f)\n', k, k*Ts, S(1), S(2), S(3));
    end
end

w.close();
sim3d.export(w, 'quadrotor_pid_mpc_3d.html');
fprintf('Wrote quadrotor_pid_mpc_3d.html\n');
