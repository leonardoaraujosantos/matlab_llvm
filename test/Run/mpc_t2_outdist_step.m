% MPC Tier-2 §3.3 — output-disturbance estimator one-tick verification.
% When obj.outdist=1, a single mpcmove tick with a non-zero measurement
% (ym = 0.3 against a zero plant state) should:
%   - estimate dist = ym - C·xp_new (after the Kalman update)
%   - subtract dist from the reference so the QP doesn't over-react
% Without outdist (Tier-1), the QP would have driven u up to track
% r=1 from y=0.3 — i.e. a much smaller move.  With outdist=1, dist
% absorbs the 0.3 offset so the controller sees an effective ref of
% (1 - 0.3) = 0.7 ABOVE the current state estimate, but the Kalman
% update already absorbed the 0.3 in its innovation = (ym-dist)-Cxp,
% so the steady picture is: the controller tracks r despite a
% constant additive output disturbance.
%
% Verify by comparing the same scenario with and without outdist.

A = [0.8, 0.0; 0.0, 0.9];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

% --- Tier-1 behavior (outdist = 0).
obj1 = mpc(sys_d, 5, 2);
obj1.umax = [10];
obj1.umin = [-10];
st1 = mpcstate(2, 1, 1);
ym = [0.3];
r  = [1];
u_no = mpcmove(obj1, st1, ym, r);
fprintf('u (outdist=0): %.4f\n', u_no(1));
fprintf('dist (must stay 0): %.4f\n', st1.Dist(1, 1));

% --- Tier-2 behavior (outdist = 1).
obj2 = mpc(sys_d, 5, 2);
obj2.umax = [10];
obj2.umin = [-10];
obj2.outdist = 1;
st2 = mpcstate(2, 1, 1);
u_yes = mpcmove(obj2, st2, ym, r);
fprintf('u (outdist=1): %.4f\n', u_yes(1));
% On the FIRST tick, dist = ym - C·xp_new ≈ ym - L·ym (Kalman gain
% already absorbed part of ym).  Multi-tick convergence drives dist
% to its true offset; a single-tick value of ~0.16 is the expected
% partial estimate.
fprintf('dist (one-tick partial estimate): %.4f\n', st2.Dist(1, 1));
