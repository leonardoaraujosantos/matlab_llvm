% MPC Tier-4 §5.7 — Finite Control Set MPC for a surge tank with
% a binary valve (User's Guide §2.28 simplified).  The MV is the
% on/off valve state, restricted to {0, 1}.  The MPC enumerates
% both branches each tick and picks the lower-cost one.

% Tank dynamics: level decays with leakage rate 0.05 per tick,
% gains flow rate 0.2 when the valve is open.
A = [0.95];
B = [0.2];
C = [1];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

mpcobj = mpc(sys_d, 5, 2);
mpcobj.umax = [1];
mpcobj.umin = [0];
mpcobj.mv_binary = [1];        % single binary MV (the valve)

st = mpcstate(1, 1, 1);

% Sequence of measurements + references — verify the FCS MPC picks
% the right branch.
ym1 = [0.2];
r   = [0.8];                    % drive level up to 0.8
u1  = mpcmoveFinite(mpcobj, st, ym1, r);
fprintf('y=0.2, r=0.8 → u = %.4f (should be 1, valve open)\n', u1(1, 1));

% After valve opens, level rises; check next tick.
ym2 = [0.4];
u2  = mpcmoveFinite(mpcobj, st, ym2, r);
fprintf('y=0.4, r=0.8 → u = %.4f\n', u2(1, 1));

% Now reverse: r=0 should close the valve.
st2 = mpcstate(1, 1, 1);
st2.Plant = [0.5];
ym3 = [0.5];
r2  = [0];
u3  = mpcmoveFinite(mpcobj, st2, ym3, r2);
fprintf('y=0.5, r=0 → u = %.4f (should be 0, valve closed)\n', u3(1, 1));
