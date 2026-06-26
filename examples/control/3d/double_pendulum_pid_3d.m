% examples/control/3d/double_pendulum_pid_3d.m
% --------------------------------------------------------------------
% DOUBLE inverted pendulum on a cart, "PID" stabilization, in 3-D.
%
%   HONEST LIMITATION
%   -----------------
%   A double inverted pendulum on a cart has TWO unstable modes but only ONE
%   actuator (the cart force). It is genuinely underactuated, and a bank of
%   independent single-loop PID controllers CANNOT robustly stabilize it —
%   each link's PID fights the other through the shared input. Classical
%   single-loop PID is the right tool for the single cart-pole
%   (see cartpole_pid_3d.m), not for this plant.
%
%   What this example actually does, therefore, is a best-effort hand-tuned
%   FULL-STATE PD law: proportional + derivative feedback on the cart position
%   and BOTH link angles, plus a small integral trim on the cart position to
%   cancel drift. That is effectively a manually-chosen state-feedback gain
%   (compare double_pendulum_lqr_3d.m / double_pendulum_place_3d.m, which
%   synthesise the gain from (A,B) instead of guessing it). It is included for
%   completeness of the PID / LQR / pole-placement trio, with this caveat.
%
%   Plant : same nonlinear two-link cart model as double_pendulum_lqr_3d.m.
%           X = [x; th1; th2; xdot; th1dot; th2dot].
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < double_pendulum_pid_3d.m
%     xdg-open double_pendulum_pid_3d.html

% ---------- Plant parameters -----------------------------------------
M  = 1.0;  m1 = 0.3;  m2 = 0.3;  L1 = 0.5;  L2 = 0.5;  g = 9.81;

% ---------- Hand-tuned full-state PD gains ---------------------------
% u = gx*x + gxd*xd + g1*th1 + g1d*th1d + g2*th2 + g2d*th2d + gi*int(x)
% (signs reflect the linkage coupling; the upper link needs opposite-sign
%  feedback to the lower one through the single shared input).
gx  = -5;    gxd = -11;       % cart position / velocity (PD)
g1  = 280;   g1d = 14;        % lower link angle / rate
g2  = -365;  g2d = -60;       % upper link angle / rate
gi  = -1.5;                   % slow integral trim on cart position

% ---------- 3-D scene -------------------------------------------------
cartH = 0.20;  hinge = cartH / 2;
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [6 6 1];  ground.Color = [0.16 0.17 0.20];
w.add(ground);

cart = sim3d.Actor('cart', 'box');
cart.Size = [0.5 0.3 cartH];  cart.Color = [0.85 0.55 0.20];
w.add(cart);

hub0 = sim3d.Actor('hub0', 'box');  hub0.Color = [0.85 0.85 0.20];
w.add(hub0);  hub0.setParent(cart);  hub0.Size = [0.06 0.06 0.06];
hub0.Translation = [0 0 hinge];

link1 = sim3d.Actor('link1', 'box');
link1.Size = [0.04 0.04 L1];  link1.Color = [0.90 0.30 0.25];
w.add(link1);  link1.setParent(hub0);  link1.Translation = [0 0 L1/2];

hub1 = sim3d.Actor('hub1', 'box');  hub1.Color = [0.85 0.85 0.20];
w.add(hub1);  hub1.setParent(hub0);  hub1.Size = [0.05 0.05 0.05];
hub1.Translation = [0 0 L1];

link2 = sim3d.Actor('link2', 'box');
link2.Size = [0.04 0.04 L2];  link2.Color = [0.70 0.40 0.85];
w.add(link2);  link2.setParent(hub1);  link2.Translation = [0 0 L2/2];

bob = sim3d.Actor('bob', 'sphere');  bob.Color = [0.95 0.50 0.20];
w.add(bob);  bob.setParent(hub1);  bob.Scale = [0.10 0.10 0.10];
bob.Translation = [0 0 L2];

w.open();

% ---------- Simulation ------------------------------------------------
Ts = 0.02;  N = 300;  n_sub = 6;  dt = Ts / n_sub;
X = [0; 0.06; 0.08; 0; 0; 0];     % small perturbation (tighter basin than LQR)
i_x = 0;                          % integral of cart position

fprintf('Double pendulum PID / full-state PD (3-D): Ts=%.3f, %d steps.\n', Ts, N);

for k = 1:N
    for j = 1:n_sub
        i_x = i_x + X(1) * dt;
        u = gx*X(1) + gxd*X(4) + g1*X(2) + g1d*X(5) + g2*X(3) + g2d*X(6) + gi*i_x;

        th1=X(2); th2=X(3); t1d=X(5); t2d=X(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k1=[X(4); t1d; t2d; a(1); a(2); a(3)];

        Xa=X+(dt/2)*k1; th1=Xa(2); th2=Xa(3); t1d=Xa(5); t2d=Xa(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k2=[Xa(4); t1d; t2d; a(1); a(2); a(3)];

        Xb=X+(dt/2)*k2; th1=Xb(2); th2=Xb(3); t1d=Xb(5); t2d=Xb(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k3=[Xb(4); t1d; t2d; a(1); a(2); a(3)];

        Xc=X+dt*k3; th1=Xc(2); th2=Xc(3); t1d=Xc(5); t2d=Xc(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k4=[Xc(4); t1d; t2d; a(1); a(2); a(3)];

        X = X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    end

    dth = X(3) - X(2);
    cart.Translation = [X(1) 0 hinge];
    hub0.Rotation = [0 0 X(2)];
    hub1.Rotation = [0 0 dth];
    w.run(Ts);

    if mod(k, 30) == 0
        fprintf('  step %3g  t=%4.2f  x=%+6.3f  th1=%+6.2f  th2=%+6.2f deg\n', ...
                k, k*Ts, X(1), X(2)*180/pi, X(3)*180/pi);
    end
end

w.close();
sim3d.export(w, 'double_pendulum_pid_3d.html');
fprintf('settled: th1=%.3f, th2=%.3f deg; wrote double_pendulum_pid_3d.html\n', ...
        X(2)*180/pi, X(3)*180/pi);
