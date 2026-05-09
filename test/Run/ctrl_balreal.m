% Tier 4 (CST roadmap §6.1) — balanced realization.
% T = balreal_T(A, B, C) returns the similarity transform such that
% the realization (T^{-1} A T, T^{-1} B, C T) has Wc = Wo = diag(HSV).
%
% This is the structural primitive for balanced model reduction:
% drop the columns/rows of the balanced realization corresponding
% to the smallest HSVs.

% --- 1. 2-state stable plant.
A = [0-1, 0; 0, 0-2];
B = [1; 1];
C = [1, 1];

T  = balreal_T(A, B, C);
Ti = inv(T);
Ab = Ti * A * T;
Bb = Ti * B;
Cb = C  * T;

% Hankel singular values (descending) — the diagonal of the balanced
% gramians.
H = hsvd(A, B, C);
disp('Hankel singular values:');
disp(H);

% Verify the balanced controllability gramian = diag(HSV).
Wcb = gram_c(Ab, Bb);
disp('balanced Wc (should equal diag(HSV)):');
disp(Wcb);

% Verify the balanced observability gramian = diag(HSV).
Wob = gram_o(Ab, Cb);
disp('balanced Wo (should equal diag(HSV)):');
disp(Wob);

% Wcb and Wob must equal each other.
diff = Wcb - Wob;
disp('Wcb - Wob (should be ~0):');
disp(diff);
