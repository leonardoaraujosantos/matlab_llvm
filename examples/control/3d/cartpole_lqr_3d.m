% examples/control/3d/cartpole_lqr_3d.m
% --------------------------------------------------------------------
% Inverted pendulum on a cart (cart-pole), stabilized by a state-space
% LQR and visualised in 3-D.
%
%   Plant   : nonlinear cart-pole, single force u on the cart, point-mass
%             pole of mass m at the tip of a massless rod of length L.
%             State  X = [x; xdot; theta; thetadot]  (theta from upright).
%   Design  : linearize about theta = 0, wrap in ss(), then K = lqr(A,B,Q,R).
%             The SAME gain drives the *nonlinear* plant (design-linear,
%             test-nonlinear — the honest controls workflow).
%   Control : u = -K x  (full-state feedback; all states assumed measured).
%   Viewer  : sim3d records one keyframe per control step; the pole is a thin
%             box parented to a pivot "hub" on the cart top, so the base stays
%             hinged at the cart while the tip swings.
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < cartpole_lqr_3d.m
%     xdg-open cartpole_lqr_3d.html

% ---------- Plant parameters -----------------------------------------
M = 1.0;      % cart mass            [kg]
m = 0.2;      % pole (tip) mass      [kg]
L = 0.6;      % rod length           [m]
g = 9.81;     % gravity              [m/s^2]

% ---------- Linearized design model ----------------------------------
% Mass matrix at upright: [M+m, m L; 1, L].  Accelerations are
%   [xdd; thetadd] = Minv * ( [0; g*theta] + [1;0]*u ).
Mq0  = [M + m, m*L; 1, L];
Minv = inv(Mq0);
a_xt = Minv(1,2) * g;     % d(xdd)/d(theta)
a_tt = Minv(2,2) * g;     % d(thetadd)/d(theta)
b_x  = Minv(1,1);         % d(xdd)/du
b_t  = Minv(2,1);         % d(thetadd)/du

A = [0 1 0 0;
     0 0 a_xt 0;
     0 0 0 1;
     0 0 a_tt 0];
B = [0; b_x; 0; b_t];
C = eye(4);
D = zeros(4, 1);
sys = ss(A, B, C, D);

disp('open-loop poles (one in the RHP — the pole falls):');
disp(real(eig(A)));

% ---------- LQR design -----------------------------------------------
% Penalise pole angle hard, cart position lightly, effort modestly.
Q = diag([8, 1, 120, 5]);
R = 0.5;
K = lqr(A, B, Q, R);
disp('LQR gain K:');
disp(K);
disp('closed-loop poles (all in the LHP):');
disp(real(eig(A - B*K)));

% ---------- 3-D scene -------------------------------------------------
cartH = 0.20;             % cart height; top face (the hinge) at cartH/2
hinge = cartH / 2;

w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [6 6 1];
ground.Color = [0.16 0.17 0.20];
w.add(ground);

cart = sim3d.Actor('cart', 'box');
cart.Size = [0.5 0.3 cartH];
cart.Color = [0.20 0.55 0.95];
w.add(cart);

% Pivot hub: a tiny box joint on the cart top. Rotating the hub swings every
% child about the hinge, so the pole pivots about its base (not its centre).
% Sized via Size (a box), NOT Scale — a parent's Scale is inherited by its
% children, so scaling the hub would shrink the pole and bob with it.
hub = sim3d.Actor('hub', 'box');
hub.Color = [0.85 0.85 0.20];
hub.Size = [0.06 0.06 0.06];
w.add(hub);
hub.setParent(cart);
hub.Translation = [0 0 hinge];

% Pole: thin box of length L standing along +Z, base at the hub.
pole = sim3d.Actor('pole', 'box');
pole.Size = [0.04 0.04 L];
pole.Color = [0.90 0.30 0.25];
w.add(pole);
pole.setParent(hub);
pole.Translation = [0 0 L/2];

% Tip bob (visualises the point mass).
bob = sim3d.Actor('bob', 'sphere');
bob.Color = [0.95 0.50 0.20];
w.add(bob);
bob.setParent(hub);
bob.Scale = [0.12 0.12 0.12];
bob.Translation = [0 0 L];

w.open();

% ---------- Simulation ------------------------------------------------
Ts    = 0.02;             % control / frame period [s]  (50 FPS)
N     = 250;              % steps (5 s)
n_sub = 4;                % RK4 sub-steps per control step
dt    = Ts / n_sub;

X = [0; 0; 0.20; 0];      % start ~11.5 deg off upright

fprintf('Cart-pole LQR (3-D): Ts=%.3f, %d steps.\n', Ts, N);

for k = 1:N
    for j = 1:n_sub
        u = -K * X;

        % --- RK4 on the nonlinear cart-pole ---------------------------
        th = X(3); thd = X(4); c = cos(th); s = sin(th);
        det1 = L * ((M + m) - m*c*c);
        f1 = u + m*L*s*thd*thd;  f2 = g*s;
        k1 = [X(2); (L*f1 - m*L*c*f2)/det1; X(4); (-c*f1 + (M+m)*f2)/det1];

        Xa = X + (dt/2)*k1;
        th = Xa(3); thd = Xa(4); c = cos(th); s = sin(th);
        det2 = L * ((M + m) - m*c*c);
        f1 = u + m*L*s*thd*thd;  f2 = g*s;
        k2 = [Xa(2); (L*f1 - m*L*c*f2)/det2; Xa(4); (-c*f1 + (M+m)*f2)/det2];

        Xb = X + (dt/2)*k2;
        th = Xb(3); thd = Xb(4); c = cos(th); s = sin(th);
        det3 = L * ((M + m) - m*c*c);
        f1 = u + m*L*s*thd*thd;  f2 = g*s;
        k3 = [Xb(2); (L*f1 - m*L*c*f2)/det3; Xb(4); (-c*f1 + (M+m)*f2)/det3];

        Xc = X + dt*k3;
        th = Xc(3); thd = Xc(4); c = cos(th); s = sin(th);
        det4 = L * ((M + m) - m*c*c);
        f1 = u + m*L*s*thd*thd;  f2 = g*s;
        k4 = [Xc(2); (L*f1 - m*L*c*f2)/det4; Xc(4); (-c*f1 + (M+m)*f2)/det4];

        X = X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    end

    % --- Record one keyframe -----------------------------------------
    cart.Translation = [X(1) 0 hinge];
    hub.Rotation = [0 0 X(3)];
    w.run(Ts);

    if mod(k, 25) == 0
        fprintf('  step %3g  t=%4.2f  x=%+6.3f  theta=%+7.3f deg\n', ...
                k, k*Ts, X(1), X(3)*180/pi);
    end
end

w.close();
sim3d.export(w, 'cartpole_lqr_3d.html');
fprintf('pole settled to %.3f deg; wrote cartpole_lqr_3d.html\n', X(3)*180/pi);
