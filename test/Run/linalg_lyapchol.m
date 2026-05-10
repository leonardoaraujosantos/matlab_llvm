% lyapchol — Cholesky factor of the controllability gramian
% (Tier-1.4 follow-on of CST roadmap §2.4).
% R = lyapchol(A, B) returns upper-triangular R with R'·R = Wc, where
% Wc solves A·Wc + Wc·A' + B·B' = 0. Used by balanced-truncation model
% reduction to dodge the squaring-of-condition-number that hits a
% chol(Wc) round trip.

% --- 1. 1×1 closed form: A=-2, B=1.
%   Wc satisfies -2·Wc + Wc·(-2) + 1 = 0  →  Wc = 1/4.
%   R = sqrt(Wc) = 0.5.
A1 = -2;
B1 = 1;
R1 = lyapchol(A1, B1);
disp(R1(1,1));         % 0.5

% --- 2. Verify the round-trip identity R' R = Wc on a stable 2×2.
A2 = [-1 2; 0 -3];
B2 = [1; 1];
R2 = lyapchol(A2, B2);
RtR = R2' * R2;
% lyap-derived Wc.
Wc = lyap(A2, B2 * B2');
% Print the 4 entries of Wc and RtR; they should match within ~1e-12.
disp(RtR(1,1) - Wc(1,1));
disp(RtR(1,2) - Wc(1,2));
disp(RtR(2,1) - Wc(2,1));
disp(RtR(2,2) - Wc(2,2));

% --- 3. Mass-spring-damper plant. Lightly damped, B = [0; 1] feeds
% velocity. R must be upper-triangular (R(2,1) = 0) and positive on
% the diagonal.
M = 1; K = 1; D = 0.1;
A3 = [0 1; -K/M -D/M];
B3 = [0; 1];
R3 = lyapchol(A3, B3);
disp(R3(2,1));         % 0  (upper triangular)
% R(1,1), R(2,2) are positive — print directly to confirm sign.
disp(R3(1,1));
disp(R3(2,2));
