% Tier 3 — discrete-time stability + H₂ norm.
% isstable_d(A): 1 if all |eig(A)| < 1, else 0. Marginal eigenvalues
% on the unit circle fail per MATLAB convention.
% norm_h2_d(A, B, C, D) = sqrt(trace(D D') + trace(C Wc C')) with
% Wc = dlyap(A, B B'); +Inf if A not Schur-stable.

% --- 1. Schur-stable diagonal plant.
A1 = [0.5, 0; 0, 0.7];
disp('isstable_d(diag(0.5, 0.7)):');
disp(isstable_d(A1));   % 1

% --- 2. Unstable plant (a > 1).
A2 = [1.5, 0; 0, 0.5];
disp('isstable_d(diag(1.5, 0.5)):');
disp(isstable_d(A2));   % 0

% --- 3. Marginal — eig on unit circle.
A3 = [0, 1; 0-1, 0];
disp('isstable_d([0 1; -1 0]) (eigs ±i):');
disp(isstable_d(A3));   % 0

% --- 4. SISO 1st-order discrete H₂ closed form. a = 0.5, b = c = 1,
% D = 0. Wc = dlyap(0.5, 1) = 4/3; ||G||_2 = 2/sqrt(3).
A4 = [0.5];
B4 = [1];
C4 = [1];
D4 = [0];
disp('discrete H2 norm (closed form 2/sqrt(3) = 1.154701):');
disp(norm_h2_d(A4, B4, C4, D4));

% --- 5. D-only contribution. a = b = c = 0, D = 3 → ||G||_2 = 3.
disp('D-only H2 norm (||G||_2 = |D| = 3):');
disp(norm_h2_d([0], [0], [0], [3]));

% --- 6. Unstable plant returns +Inf.
disp('unstable plant H2 norm (must be +Inf):');
disp(norm_h2_d([1.5], [1], [1], [0]));

% --- 7. Discretise a continuous mass-spring-damper, compare H2 norms.
wn = 3.0; zeta = 0.5;
Ac = [0, 1; 0-wn*wn, 0-2*zeta*wn];
Bc = [0; 1];
Cc = [1, 0];
Dc = [0];
[Ad, Bd] = c2d(Ac, Bc, 0.1);
disp('continuous ||G||_2:');
disp(norm_h2(Ac, Bc, Cc));
disp('discretised (Ts = 0.1) ||G||_2 (small Ts → approaches continuous):');
disp(norm_h2_d(Ad, Bd, Cc, Dc));
