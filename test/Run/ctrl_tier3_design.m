% Tier-3 follow-ons — state-space design surface.
%
% Backs matrix-arg runtime entries (matlab_lqr_5 / matlab_dlqr_5
% for cross-term LQR, matlab_lqry_ss for output-weighted LQR,
% matlab_place / acker alias) and the model-object short forms
% (ctrb(sys), obsv(sys), gram(sys, 'c'/'o'), norm(sys, 2),
% lqry(sys, Q, R)).
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_sys_short.

A = [-1 0; 0 -2];
B = [1; 1];
C = [1 0];
D = [0];
sys = ss(A, B, C, D);

% --- §4.4 ctrb(sys) / obsv(sys) / gram(sys, 'c' | 'o')
disp(ctrb(sys));
disp(obsv(sys));
disp(gram(sys, 'c'));
disp(gram(sys, 'o'));

% --- §4.5 norm(sys) / norm(sys, 2) — H₂ system norm.
disp(norm(sys));
disp(norm(sys, 2));

% --- §4.1 5-arg cross-term lqr(A, B, Q, R, N).
Q = [1 0; 0 1];
R = [1];
N = [0.1; 0.1];
K5 = lqr(A, B, Q, R, N);
disp(K5);

% --- §4.1 lqry(sys, Q, R) — output-weighted LQR.
%   For diagonal A, B = [1;1], C = [1, 0], Q_y = 1, R = 1:
%   Q_x = C'·Q·C = diag(1, 0); only state-1 weighted.
%   CARE on state-1: K1 = -1 + sqrt(2) ≈ 0.414; K2 = 0.
Klqry = lqry(sys, [1], [1]);
disp(Klqry);

% --- §4.3 acker(A, B, p) — alias of place. Both pick a SISO gain
% that places closed-loop poles at p.
Ka = acker(A, B, [-3; -4]);
disp(Ka);
