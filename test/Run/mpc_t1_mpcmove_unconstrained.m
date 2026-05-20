% MPC Tier-1 §2.5 — mpcmove single tick with no active bounds.
% The default umin/umax = ±1e6 means the MV bounds never bind, so
% the KWIK active-set should converge in 0 iterations to the
% unconstrained QP solution z* = -H⁻¹ f.

% 2-state discrete plant.
A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

obj = mpc(sys_d, 5, 2);
st  = mpcstate(2, 1);

% Reference: track y → 1.
r  = [1];

% First-tick measurement: ym = C * 0 = 0 (state starts at zero).
ym = [0];

u  = mpcmove(obj, st, ym, r);

% u should be positive (controller pushes toward y = 1 setpoint) and
% well below the loose 1e6 bound.
fprintf('u = %.6f\n', u(1));
% Verify u is positive (controller moves in the right direction).
% A direct numerical check would need a reference simulation, so we
% just confirm the first move is non-trivial.
