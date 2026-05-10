% 5-arg `care(A, B, Q, R, S)` — Riccati with state-input cross term
% (Tier-1.5 follow-on of CST roadmap §2.5). Reduces to the 4-arg form
% via A_hat = A − B·R⁻¹·S' and Q_hat = Q − S·R⁻¹·S'.

% --- 1. Cross-term S = 0 must give the same X as 4-arg care.
A = [0 1; 0 0];
B = [0; 1];
Q = eye(2);
R = 1;
S0 = [0; 0];
X0 = care(A, B, Q, R, S0);
Xref = care(A, B, Q, R);
disp(round((X0(1,1) - Xref(1,1))*1e10)/1e10);     % 0
disp(round((X0(1,2) - Xref(1,2))*1e10)/1e10);     % 0
disp(round((X0(2,2) - Xref(2,2))*1e10)/1e10);     % 0

% --- 2. Non-zero S — verify that the reduced problem solution
% satisfies the original 5-arg Riccati residual:
%   A'X + XA - (XB + S)R⁻¹(B'X + S') + Q = 0
S1 = [1; 1];
X1 = care(A, B, Q, R, S1);
% Compute residual.
Rinv = 1 / R;
M = X1 * B + S1;
res = A' * X1 + X1 * A - M * Rinv * M' + Q;
% Each entry should be ~0 (within ~1e-9 — the matrix-sign Newton
% iteration converges to that residual).
disp(round(res(1,1)*1e8)/1e8);
disp(round(res(1,2)*1e8)/1e8);
disp(round(res(2,2)*1e8)/1e8);

% --- 3. Discrete companion: dare(A, B, Q, R, S).
Ad = [0.5 0; 0 0.7];
Bd = eye(2);
Qd = eye(2);
Rd = eye(2);
Sd = zeros(2, 2);
Xd0 = dare(Ad, Bd, Qd, Rd, Sd);
Xd_ref = dare(Ad, Bd, Qd, Rd);
disp(round((Xd0(1,1) - Xd_ref(1,1))*1e10)/1e10);  % 0
disp(round((Xd0(2,2) - Xd_ref(2,2))*1e10)/1e10);  % 0
