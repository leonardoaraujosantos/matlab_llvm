% Tier 3 (CST roadmap §4.2-ish — design primitive).
% ctrb / obsv / place: structural-rank companions to gram_c / gram_o
% plus SISO Ackermann pole placement.
%   ctrb(A, B) = [B, A B, ..., A^{n-1} B]    is n x (n*m).
%   obsv(A, C) = [C; C A; ...; C A^{n-1}]    is (p*n) x n.
%   place(A, B, P) returns K s.t. eig(A - B K) = P (SISO; B is n x 1).
% Multi-input pole placement uses Kautsky-Nichols-Van Dooren and is
% deferred; SISO Ackermann is the universally-taught form.

% --- 1. Double integrator: A = [0 1; 0 0], B = [0; 1].
% Co = [B, A B] = [[0 1]; [1 0]] (full rank, controllable).
% Ob with C = [1 0]: [[1 0]; [0 1]] = I  (full rank, observable).
A = [0 1; 0 0];
B = [0; 1];
C = [1, 0];

Co = ctrb(A, B);
disp('controllability matrix Co:');
disp(Co);

Ob = obsv(A, C);
disp('observability matrix Ob:');
disp(Ob);

% --- 2. SISO place puts double-integrator poles at {-1, -2}.
% Closed-form (Ackermann): K = [2, 3].
P = [0-1; 0-2];
K = place(A, B, P);
disp('place(A, B, P) for double integrator (closed form K = [2, 3]):');
disp(K);

% Closed-loop spectrum must equal P (sorted ascending real part).
Acl = A - B * K;
disp('eig(A - B K):');
disp(real(eig(Acl)));

% --- 3. 3x3 controllable companion with desired poles {-1, -2, -3}.
% A2 has open-loop eigenvalues at the roots of  s^3 + 6 s^2 - 11 s - 6,
% which is a non-canonical polynomial; place reassigns them to the
% desired locations.
A2 = [0 1 0; 0 0 1; 6 11 0-6];
B2 = [0; 0; 1];
P2 = [0-1; 0-2; 0-3];
K2 = place(A2, B2, P2);
disp('3-state place gain K2:');
disp(K2);

Acl2 = A2 - B2 * K2;
disp('eig(A - B K) (should be {-3, -2, -1}):');
disp(real(eig(Acl2)));
