% examples/mpc/pendulum_nlmpc.m — Tier-5 headline (Nonlinear MPC).
%
% Damped pendulum:
%   ẋ1 = x2
%   ẋ2 = -sin(x1) - 0.1·x2 + u
%
% Build an `nlmpc(2, 1, 1)` controller and drive the angle from a
% displaced initial state back to the down-equilibrium.  StateFcn
% is supplied as an anonymous function handle with the packed
% `zxu = [x; u]` single-arg signature.
%
% Tier-5 simplifications: Forward Euler integration, default
% tracking cost.  RK4 / CustomCostFcn / multistage NMPC are
% Tier-5 carve-downs.

nlobj = nlmpc(2, 1, 1);
nlobj.Ts = 0.1;
nlobj.p  = 10;
nlobj.m  = 3;
nlobj.umax = [5];
nlobj.umin = [-5];

state_fn = @(zxu) [zxu(2, 1); 0-sin(zxu(1, 1)) - 0.1*zxu(2, 1) + zxu(3, 1)];

x = [0.2; 0];
u_prev = [0];
r = [0];

u = nlmpcmove(nlobj, x, u_prev, r, state_fn);
fprintf('NMPC first move on pendulum at x=[0.2; 0]: u = %.4f\n', u(1, 1));
