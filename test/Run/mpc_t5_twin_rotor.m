% MPC Tier-5 — MIMO nonlinear MPC (twin-rotor / 2-DOF helicopter).
% Numeric-only gating copy of examples/mpc/twin_rotor_nlmpc.m: same
% nlmpc(4,2,2) controller + RK4 plant, shortened to 40 steps, single
% setpoint, no plots.  Verifies MIMO nonlinear MPC (nu=2, ny=2, nx=4)
% coordinates two cross-coupled rotors to track pitch and yaw.

nlobj = nlmpc(4, 2, 2);
nlobj.Ts = 0.1;
nlobj.p  = 12;
nlobj.m  = 3;
nlobj.umax = [4; 4];
nlobj.umin = [-4; -4];
nlobj.Wy  = [3; 3];
nlobj.Wdu = [0.1; 0.1];

state_fn = @(zxu) [zxu(3,1); zxu(4,1); (0.12*zxu(5,1) + 0.02*zxu(6,1) - 0.02*zxu(3,1) - 0.4*sin(zxu(1,1)))/0.05; (0.10*zxu(6,1) + 0.03*zxu(5,1) - 0.02*zxu(4,1))/0.04];

Ts = 0.1;
n_inner = 5;
dt = Ts / n_inner;
N = 40;

S = zeros(4, 1);
u_prev1 = 0;
u_prev2 = 0;
r = [0.3; 0.5];

log_p = zeros(1, N);
log_y = zeros(1, N);

for k = 1:N
    u_prev = [u_prev1; u_prev2];
    u = nlmpcmove(nlobj, S, u_prev, r, state_fn);
    uu = zeros(2, 1);
    uu(1) = u(1, 1);
    uu(2) = u(2, 1);
    u1 = uu(1);
    u2 = uu(2);
    u_prev1 = u1;
    u_prev2 = u2;

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
    log_p(k) = S(1);
    log_y(k) = S(2);
end

fprintf('twin-rotor MIMO nlmpc: %g steps\n', N);
fprintf('pitch at k=10: %.3f  (ref 0.3)\n', log_p(10));
fprintf('yaw   at k=10: %.3f  (ref 0.5)\n', log_y(10));
fprintf('pitch settle  : %.3f  (ref 0.3)\n', log_p(40));
fprintf('yaw settle    : %.3f  (ref 0.5)\n', log_y(40));
fprintf('final rotors  : u1=%.3f u2=%.3f\n', u1, u2);
