% Tier 3 — model characterization triad demo.
%   isstable(A): Hurwitz check.
%   damp(A):     per-pole [wn, zeta] table.
%   hsvd(A,B,C): Hankel singular values — diagnostic for model
%                reduction (small HSVs → states with little I/O impact).
%
% These three sit on top of the gram_c/gram_o (Tier 1.4 lyap) +
% non-symmetric eig (Tier 1.1) primitives that landed earlier in the
% CST roadmap.

% --- 1. Mass-spring-damper, lightly damped.
% xdot = [0 1; -wn^2 -2*zeta*wn] x + [0; 1] u, y = [1 0] x.
wn = 3.0; zeta = 0.05;
A = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B = [0; 1];
C = [1, 0];

disp('open-loop isstable (lightly-damped Hurwitz expected → 1):');
disp(isstable(A));

disp('open-loop damp [wn, zeta]:');
disp(damp(A));

disp('hsvd (single-mode plant — both HSVs nonzero):');
disp(hsvd(A, B, C));

% --- 2. After LQR design, closed-loop should be stiffer (higher zeta).
Q  = [1 0; 0 1];
R  = [1];
K  = lqr(A, B, Q, R);
Ac = A - B * K;
disp('closed-loop isstable (LQR makes Hurwitz → 1):');
disp(isstable(Ac));

disp('closed-loop damp (zeta should be larger than open-loop):');
disp(damp(Ac));

disp('closed-loop hsvd:');
disp(hsvd(Ac, B, C));

% --- 3. A redundant 4-state plant — appended a near-uncontrollable mode.
% A4 = blkdiag(A, -10) extended; the slow well-coupled modes dominate
% the I/O behaviour, so hsvd should expose the small-HSV state as the
% target for `balred`.
A4 = [0,         1,        0,    0;
      0-wn*wn,  -2*zeta*wn, 0,    0;
      0,         0,         0-10, 0;
      0,         0,         0,    0-20];
B4 = [0; 1; 0.001; 0.001];   % small input coupling to the fast modes
C4 = [1, 0, 0.01, 0.01];
disp('redundant 4-state plant — hsvd should rank-order the modes:');
disp(hsvd(A4, B4, C4));
