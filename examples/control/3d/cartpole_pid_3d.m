% examples/control/3d/cartpole_pid_3d.m
% --------------------------------------------------------------------
% Inverted pendulum on a cart, stabilized by CLASSICAL PID and shown in 3-D.
%
%   Primary loop : a PID on the pole angle theta (proportional + integral +
%                  derivative) supplies most of the stabilizing force.
%   Trim loop    : a slow proportional-derivative term on the cart position
%                  recentres the cart, which a bare angle-PID would let drift.
%   Plant        : nonlinear cart-pole (same model as cartpole_lqr_3d.m).
%
%   Unlike the LQR / pole-placement siblings this controller is tuned by hand
%   in classical-control fashion, not synthesised from (A, B). The single
%   inverted pendulum is the canonical plant where a hand-tuned PID works
%   well; contrast with double_pendulum_pid_3d.m, where it does not.
%
% Run it interpreted, then open the HTML:
%     matlabc -repl < cartpole_pid_3d.m
%     xdg-open cartpole_pid_3d.html

% ---------- Plant parameters -----------------------------------------
M = 1.0; m = 0.2; L = 0.6; g = 9.81;

% ---------- PID gains (hand-tuned) -----------------------------------
Kp_th = 60;   Ki_th = 8;   Kd_th = 12;     % angle PID (primary)
Kp_x  = 3.5;  Kd_x  = 5.5;                 % cart-centering trim (PD)

% ---------- 3-D scene -------------------------------------------------
cartH = 0.20;  hinge = cartH / 2;
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [6 6 1];  ground.Color = [0.16 0.17 0.20];
w.add(ground);

cart = sim3d.Actor('cart', 'box');
cart.Size = [0.5 0.3 cartH];  cart.Color = [0.85 0.55 0.20];
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
X = [0; 0; 0.18; 0];      % start ~10.3 deg off upright
i_th = 0;                 % integral of theta

fprintf('Cart-pole PID (3-D): Ts=%.3f, %d steps.\n', Ts, N);

for k = 1:N
    for j = 1:n_sub
        th = X(3); thd = X(4);
        i_th = i_th + th * dt;
        u = Kp_th*th + Ki_th*i_th + Kd_th*thd + Kp_x*X(1) + Kd_x*X(2);

        % --- RK4 on the nonlinear cart-pole (u held over the sub-step) -
        c = cos(th); s = sin(th);
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
    hub.Rotation = [0 0 X(3)];
    w.run(Ts);

    if mod(k, 25) == 0
        fprintf('  step %3g  t=%4.2f  x=%+6.3f  theta=%+7.3f deg\n', ...
                k, k*Ts, X(1), X(3)*180/pi);
    end
end

w.close();
sim3d.export(w, 'cartpole_pid_3d.html');
fprintf('pole settled to %.3f deg; wrote cartpole_pid_3d.html\n', X(3)*180/pi);
