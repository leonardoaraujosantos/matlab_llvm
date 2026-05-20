% MPC Tier-5 §6.1/6.2 — Nonlinear MPC for a pendulum.
% Damped pendulum dynamics:
%   ẋ1 = x2
%   ẋ2 = -sin(x1) - 0.1·x2 + u
% nlmpc uses the nonlinear dynamics directly via an anonymous
% handle (single-arg with packed zxu = [x; u] vector, consistent
% with the Optim Tier-2 fmincon handle ABI).

nlobj = nlmpc(2, 1, 1);
nlobj.Ts = 0.1;
nlobj.p  = 10;
nlobj.m  = 3;
nlobj.umax = [5];
nlobj.umin = [-5];

% StateFcn as an anonymous handle: zxu(1,1) = x1 (angle), zxu(2,1) = x2
% (rate), zxu(3,1) = u.  Returns dxdt as a 2×1 column.
state_fn = @(zxu) [zxu(2, 1); 0-sin(zxu(1, 1)) - 0.1*zxu(2, 1) + zxu(3, 1)];

% Initial state: pendulum displaced 0.2 rad with zero velocity.
x = [0.2; 0];
u_prev = [0];
r = [0];      % drive angle back to 0

u = nlmpcmove(nlobj, x, u_prev, r, state_fn);
fprintf('first NMPC move u = %.4f\n', u(1, 1));
fprintf('(expected sign: negative — push angle back toward 0)\n');
