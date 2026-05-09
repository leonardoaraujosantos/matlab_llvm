% Tier 3 — H₂ system norm (continuous LTI, strictly proper).
% norm_h2(A, B, C) = sqrt(trace(C · Wc · C')) where Wc = lyap(A, B B').
% Returns +Inf if A is not Hurwitz.

% --- 1. SISO 1st-order Hurwitz: G(s) = bc / (s + a). ||G||_2 = bc/sqrt(2a).
% Pick a = 1, b = 1, c = 1 → ||G||_2 = 1/sqrt(2) ≈ 0.707107.
A = [0-1];
B = [1];
C = [1];
disp('||G||_2 for G(s) = 1/(s+1) — closed form 1/sqrt(2):');
disp(norm_h2(A, B, C));

% --- 2. Mass-spring-damper (lightly damped).
% xdot = [0 1; -wn^2 -2*zeta*wn] x + [0; 1] u, y = [1 0] x.
wn = 3.0; zeta = 0.05;
A2 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B2 = [0; 1];
C2 = [1, 0];
disp('||G||_2 for lightly-damped MSD (zeta = 0.05):');
disp(norm_h2(A2, B2, C2));

% --- 3. Same plant, more damping → smaller H₂ norm.
zeta3 = 0.5;
A3 = [0, 1; 0-wn*wn, 0-2*zeta3*wn];
disp('||G||_2 for the same plant with zeta = 0.5 (must be smaller):');
disp(norm_h2(A3, B2, C2));

% --- 4. Unstable plant returns +Inf.
A4 = [1];
B4 = [1];
C4 = [1];
disp('||G||_2 for the unstable plant a=1 (must be +Inf):');
disp(norm_h2(A4, B4, C4));

% --- 5. Similarity-invariance: same plant in different coordinates
% gives the same norm.
T = [1, 1; 0, 1];
Ti = inv(T);
disp('||G||_2 invariant under similarity transform:');
disp(norm_h2(Ti * A2 * T, Ti * B2, C2 * T));
