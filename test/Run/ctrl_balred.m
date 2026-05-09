% Tier 4 (CST roadmap §6.1) — balanced truncation.
% balred_A / balred_B / balred_C return the k-state truncated balanced
% realization. Drop states corresponding to the smallest Hankel
% singular values; the H∞ error bound is 2·sum(HSV[k+1:n]).
%
% MATLAB-faithful  [Ar, Br, Cr] = balred(A, B, C, k)  3-return shape
% is a follow-on; today users call the three single-return entries.

% --- 4-state plant: dominant 2-state mass-spring-damper plus two
% fast modes weakly coupled (small B/C entries).
A = [0,    1,     0,     0;
     0-9, 0-0.3,  0,     0;
     0,    0,    0-10,    0;
     0,    0,    0,    0-20];
B = [0; 1; 0.001; 0.001];
C = [1, 0, 0.01, 0.01];

H = hsvd(A, B, C);
disp('Hankel singular values (descending):');
disp(H);

% --- Truncate to 2 states.
Ar = balred_A(A, B, C, 2);
Br = balred_B(A, B, C, 2);
Cr = balred_C(A, B, C, 2);

disp('reduced A (2x2):');
disp(Ar);
disp('reduced B (2x1):');
disp(Br);
disp('reduced C (1x2):');
disp(Cr);

% --- Reduced realization must still be Hurwitz.
disp('isstable(Ar) (must be 1):');
disp(isstable(Ar));

% --- The balanced truncated realization should preserve the dominant
% Hankel singular values: hsvd(Ar, Br, Cr) ≈ H(1:2).
disp('hsvd(Ar, Br, Cr) — should match the top-2 HSVs of the original:');
disp(hsvd(Ar, Br, Cr));
