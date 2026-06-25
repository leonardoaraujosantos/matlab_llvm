% examples/control/3d/double_pendulum_lqr_3d.m
% --------------------------------------------------------------------
% DOUBLE inverted pendulum on a cart, stabilized by state-space LQR, in 3-D.
%
%   Plant : cart (force u) carrying a two-link chain of point masses on
%           massless rods. Angles th1, th2 are ABSOLUTE, measured from the
%           upward vertical. State
%               X = [x; th1; th2; xdot; th1dot; th2dot]  (6 states).
%           Two unstable modes, a SINGLE actuator -> genuinely underactuated.
%   Design: linearize about upright, K = lqr(A,B,Q,R) (full-state feedback),
%           then drive the *nonlinear* plant. Full state is assumed measured.
%   Viewer: a kinematic chain cart -> hub0 -> link1 -> hub1 -> link2; rotating
%           each hub swings its link about the joint, so the linkage reads as a
%           real double pendulum.
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < double_pendulum_lqr_3d.m
%     xdg-open double_pendulum_lqr_3d.html

% ---------- Plant parameters -----------------------------------------
M  = 1.0;     % cart mass        [kg]
m1 = 0.3;     % lower-link mass  [kg]
m2 = 0.3;     % upper-link mass  [kg]
L1 = 0.5;     % lower-link length [m]
L2 = 0.5;     % upper-link length [m]
g  = 9.81;

% ---------- Linearized design model (6-state) ------------------------
Mt   = M + m1 + m2;
Mq0  = [Mt,        (m1+m2)*L1, m2*L2;
        (m1+m2),   (m1+m2)*L1, m2*L2;
        1,         L1,         L2];
Minv = inv(Mq0);
Gmat = [0,0,0; 0,(m1+m2)*g,0; 0,0,g];   % gravity -> accel (small-angle)
MG = Minv * Gmat;                       % 3x3
Mb = Minv * [1; 0; 0];                  % 3x1, force -> accel

% Assemble A by block index-assignment (2x2 block matrix *literals* do not
% concatenate reliably in the interpreter; index-assignment is robust).
A = zeros(6, 6);
A(1:3, 4:6) = eye(3);
A(4:6, 1:3) = MG;
B = [zeros(3,1); Mb];

disp('open-loop poles (RHP entries — the chain falls):');
disp(real(eig(A))');

% ---------- LQR design -----------------------------------------------
Q = diag([5, 200, 200, 1, 12, 12]);
R = 0.20;
K = lqr(A, B, Q, R);
disp('LQR gain K:');
disp(K);
disp('closed-loop poles (all in the LHP):');
disp(real(eig(A - B*K))');

% ---------- 3-D scene -------------------------------------------------
cartH = 0.20;  hinge = cartH / 2;
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [6 6 1];  ground.Color = [0.16 0.17 0.20];
w.add(ground);

cart = sim3d.Actor('cart', 'box');
cart.Size = [0.5 0.3 cartH];  cart.Color = [0.20 0.55 0.95];
w.add(cart);

hub0 = sim3d.Actor('hub0', 'sphere');  hub0.Color = [0.85 0.85 0.20];
w.add(hub0);  hub0.setParent(cart);  hub0.Scale = [0.07 0.07 0.07];
hub0.Translation = [0 0 hinge];

link1 = sim3d.Actor('link1', 'box');
link1.Size = [0.04 0.04 L1];  link1.Color = [0.90 0.30 0.25];
w.add(link1);  link1.setParent(hub0);  link1.Translation = [0 0 L1/2];

hub1 = sim3d.Actor('hub1', 'sphere');  hub1.Color = [0.85 0.85 0.20];
w.add(hub1);  hub1.setParent(hub0);  hub1.Scale = [0.06 0.06 0.06];
hub1.Translation = [0 0 L1];

link2 = sim3d.Actor('link2', 'box');
link2.Size = [0.04 0.04 L2];  link2.Color = [0.30 0.70 0.40];
w.add(link2);  link2.setParent(hub1);  link2.Translation = [0 0 L2/2];

bob = sim3d.Actor('bob', 'sphere');  bob.Color = [0.95 0.50 0.20];
w.add(bob);  bob.setParent(hub1);  bob.Scale = [0.10 0.10 0.10];
bob.Translation = [0 0 L2];

w.open();

% ---------- Simulation ------------------------------------------------
Ts = 0.02;  N = 300;  n_sub = 6;  dt = Ts / n_sub;
X = [0; 0.10; 0.14; 0; 0; 0];     % small perturbation from upright

fprintf('Double pendulum LQR (3-D): Ts=%.3f, %d steps.\n', Ts, N);

for k = 1:N
    for j = 1:n_sub
        u = -K * X;

        % --- RK4 stage 1 ----------------------------------------------
        th1=X(2); th2=X(3); t1d=X(5); t2d=X(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k1=[X(4); t1d; t2d; a(1); a(2); a(3)];

        % --- RK4 stage 2 ----------------------------------------------
        Xa=X+(dt/2)*k1; th1=Xa(2); th2=Xa(3); t1d=Xa(5); t2d=Xa(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k2=[Xa(4); t1d; t2d; a(1); a(2); a(3)];

        % --- RK4 stage 3 ----------------------------------------------
        Xb=X+(dt/2)*k2; th1=Xb(2); th2=Xb(3); t1d=Xb(5); t2d=Xb(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k3=[Xb(4); t1d; t2d; a(1); a(2); a(3)];

        % --- RK4 stage 4 ----------------------------------------------
        Xc=X+dt*k3; th1=Xc(2); th2=Xc(3); t1d=Xc(5); t2d=Xc(6);
        c1=cos(th1); s1=sin(th1); c2=cos(th2); s2=sin(th2);
        c12=cos(th1-th2); s12=sin(th1-th2);
        Mq=[M+m1+m2, (m1+m2)*L1*c1, m2*L2*c2; (m1+m2)*c1, (m1+m2)*L1, m2*L2*c12; c2, L1*c12, L2];
        f=[u + (m1+m2)*L1*s1*t1d*t1d + m2*L2*s2*t2d*t2d; (m1+m2)*g*s1 - m2*L2*s12*t2d*t2d; g*s2 + L1*s12*t1d*t1d];
        a=Mq\f;  k4=[Xc(4); t1d; t2d; a(1); a(2); a(3)];

        X = X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    end

    % --- Record one keyframe (absolute angles -> chained hub rotations) -
    cart.Translation = [X(1) 0 hinge];
    dth = X(3) - X(2);
    hub0.Rotation = [0 X(2) 0];
    hub1.Rotation = [0 dth 0];               % relative, so absolute = th2
    w.run(Ts);

    if mod(k, 30) == 0
        fprintf('  step %3g  t=%4.2f  x=%+6.3f  th1=%+6.2f  th2=%+6.2f deg\n', ...
                k, k*Ts, X(1), X(2)*180/pi, X(3)*180/pi);
    end
end

w.close();
sim3d.export(w, 'double_pendulum_lqr_3d.html');
fprintf('settled: th1=%.3f, th2=%.3f deg; wrote double_pendulum_lqr_3d.html\n', ...
        X(2)*180/pi, X(3)*180/pi);
