% examples/control/3d/cartpole_place_3d.m
% --------------------------------------------------------------------
% Inverted pendulum on a cart, stabilized by POLE PLACEMENT and shown in 3-D.
%
%   The user-facing alternative to LQR: instead of a quadratic cost, the
%   engineer names *where the closed-loop poles should sit* and place(A,B,P)
%   computes the gain that puts them there.
%
%   Plant   : nonlinear cart-pole (same model as cartpole_lqr_3d.m).
%   Design  : 1) confirm controllability via rank(ctrb(A,B)) = n,
%             2) pick a stable pole set P, 3) K = place(A,B,P),
%             4) the closed-loop A - B K then has eig = P.
%   Control : u = -K x, driving the nonlinear plant.
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < cartpole_place_3d.m
%     xdg-open cartpole_place_3d.html

% ---------- Plant parameters -----------------------------------------
M = 1.0; m = 0.2; L = 0.6; g = 9.81;

% ---------- Linearized design model ----------------------------------
Mq0  = [M + m, m*L; 1, L];
Minv = inv(Mq0);
A = [0 1 0 0;
     0 0 Minv(1,2)*g 0;
     0 0 0 1;
     0 0 Minv(2,2)*g 0];
B = [0; Minv(1,1); 0; Minv(2,1)];
C = [1 0 0 0];

% ---------- Controllability / observability checks -------------------
Co = ctrb(A, B);
disp('rank(ctrb(A,B)) (must equal 4 for arbitrary placement):');
disp(rank(Co));
Ob = obsv(A, C);
disp('rank(obsv(A,C)) with C = [1 0 0 0] (measure cart position):');
disp(rank(Ob));

% ---------- Pole placement -------------------------------------------
% A well-damped fan in the LHP: two faster modes for the pole, two slower
% for the cart.
P = [-6.0, -6.5, -2.5, -3.0];
K = place(A, B, P);
disp('place gain K:');
disp(K);
disp('eig(A - B K) (must match P):');
disp(real(eig(A - B*K)));

% ---------- 3-D scene -------------------------------------------------
cartH = 0.20;  hinge = cartH / 2;
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [6 6 1];  ground.Color = [0.16 0.17 0.20];
w.add(ground);

cart = sim3d.Actor('cart', 'box');
cart.Size = [0.5 0.3 cartH];  cart.Color = [0.30 0.75 0.45];
w.add(cart);

hub = sim3d.Actor('hub', 'box');  hub.Color = [0.85 0.85 0.20];
w.add(hub);  hub.setParent(cart);  hub.Size = [0.06 0.06 0.06];
hub.Translation = [0 0 hinge];

pole = sim3d.Actor('pole', 'box');
pole.Size = [0.04 0.04 L];  pole.Color = [0.90 0.30 0.25];
w.add(pole);  pole.setParent(hub);  pole.Translation = [0 0 L/2];

bob = sim3d.Actor('bob', 'sphere');  bob.Color = [0.95 0.50 0.20];
w.add(bob);  bob.setParent(hub);  bob.Scale = [0.12 0.12 0.12];
bob.Translation = [0 0 L];

w.open();

% ---------- Simulation ------------------------------------------------
Ts = 0.02;  N = 250;  n_sub = 4;  dt = Ts / n_sub;
X = [0; 0; -0.22; 0];     % start ~ -12.6 deg off upright

fprintf('Cart-pole pole-placement (3-D): Ts=%.3f, %d steps.\n', Ts, N);

for k = 1:N
    for j = 1:n_sub
        u = -K * X;

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

    cart.Translation = [X(1) 0 hinge];
    hub.Rotation = [0 X(3) 0];
    w.run(Ts);

    if mod(k, 25) == 0
        fprintf('  step %3g  t=%4.2f  x=%+6.3f  theta=%+7.3f deg\n', ...
                k, k*Ts, X(1), X(3)*180/pi);
    end
end

w.close();
sim3d.export(w, 'cartpole_place_3d.html');
fprintf('pole settled to %.3f deg; wrote cartpole_place_3d.html\n', X(3)*180/pi);
