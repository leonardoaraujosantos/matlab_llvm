% Tier 2 — `series_ss(...)` and `parallel_ss(...)` interconnection
% primitives, matrix-arg, strictly-proper plants only.
%
%   series:   sys = sys2 * sys1   (cascade, sys2 fed by sys1's output)
%   parallel: sys = sys1 + sys2   (same input, summed outputs)

% --- 1. Two identical 1st-order plants H(s) = 1/(s+1).
A = [0-1]; B = [1]; C = [1];

% Series: H_total(s) = 1/(s+1)^2.
[As, Bs, Cs] = series_ss(A, B, C, A, B, C);
disp('series Acl ([-1, 0; 1, -1]):');
disp(As);
disp('series eig (must both be -1):');
disp(real(eig(As)));

% Parallel: H_total(s) = 2/(s+1).
[Ap, Bp, Cp] = parallel_ss(A, B, C, A, B, C);
disp('parallel Acl (blkdiag(-1, -1)):');
disp(Ap);
disp('parallel Bcl ([1; 1]):');
disp(Bp);
disp('parallel Ccl ([1, 1]):');
disp(Cp);

% --- 2. Different plants: sys1 = 1/(s+1), sys2 = 1/(s+3).
% Series: H = 1 / ((s+1)(s+3)). Eigvals stay at -1, -3.
A2 = [0-3]; B2 = [1]; C2 = [1];
[As2, Bs2, Cs2] = series_ss(A, B, C, A2, B2, C2);
disp('series with -1 -3 plants — eig (must be {-3, -1}):');
disp(real(eig(As2)));

% --- 3. 1-return form: defaults to Acl.
A_only = series_ss(A, B, C, A2, B2, C2);
disp('1-return series Acl (must match):');
disp(A_only);
