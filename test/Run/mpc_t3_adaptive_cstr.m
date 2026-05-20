% MPC Tier-3 §4.1 — adaptive MPC: mpcmoveAdaptive rebuilds the
% cached prediction matrices from a new (A, B, C) before solving.
% Verify by comparing the MV from standard mpcmove (uses original
% plant) vs. mpcmoveAdaptive (uses a different plant).

% Initial plant (slow).
A1 = [0.9, 0.0; 0.0, 0.9];
B1 = [1; 0.5];
C1 = [1, 0];
D  = [0];
sys0 = ss(A1, B1, C1, D, 0.1);

obj = mpc(sys0, 5, 2);
obj.umax = [10];
obj.umin = [-10];

st1 = mpcstate(2, 1, 1);
ym = [0];
r  = [1];
u_orig = mpcmove(obj, st1, ym, r);
fprintf('u (original plant): %.4f\n', u_orig(1, 1));

% Adaptive call with a faster plant.
A2 = [0.5, 0.0; 0.0, 0.5];
B2 = [1; 0.5];
C2 = [1, 0];
st2 = mpcstate(2, 1, 1);
u_adapt = mpcmoveAdaptive(obj, st2, A2, B2, C2, ym, r);
fprintf('u (faster plant via adaptive): %.4f\n', u_adapt(1, 1));

% Verify obj's cached A was updated by mpcmoveAdaptive.
fprintf('obj.A(1,1) (must be 0.5 after adapt): %.4f\n', obj.A(1, 1));
