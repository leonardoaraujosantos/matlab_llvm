% Tier-4 close — sminreal / modred / thiran.
%
% Matrix-arg runtime entries (matlab_sminreal_{A,B,C} structural
% minimal realisation, matlab_modred_{A,B,C} modal residualisation
% with Truncate/MatchDC, matlab_thiran_{a,b} fractional-delay
% all-pass FIR) plus class-returning model-object short forms
% sminreal(sys) and modred(sys, elim, method) that wrap the result
% back into a fresh ss(.) constructor call.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as the existing
% model-object tests.

% --- §5.1 sminreal(sys) — drops state 2 (unobservable through C).
A = [-1 0; 0 -2];
B = [1; 1];
C = [1 0];
D = [0];
sys = ss(A, B, C, D);
red = sminreal(sys);
disp(red.A);
disp(red.B);
disp(red.C);
disp(red.D);

% --- §5.1 modred(sys, elim, method) — drop state 2 from a 3-state
% diagonal plant. Truncate just drops; MatchDC applies the Schur
% complement (here it's the same because A is diagonal so the
% A12/A21 cross terms are zero).
A3 = [-1 0 0; 0 -10 0; 0 0 -2];
B3 = [1; 1; 1];
C3 = [1 1 1];
D3 = [0];
sys3 = ss(A3, B3, C3, D3);

red_t = modred(sys3, [2], 'Truncate');
disp(red_t.A);
disp(red_t.B);
disp(red_t.C);

red_dc = modred(sys3, [2], 'MatchDC');
disp(red_dc.A);

% --- §5.3 thiran(D, n) — fractional-delay all-pass FIR.
% For D = 1.5, n = 3, the coefficient vectors are mirror-symmetric.
[b, a] = thiran(1.5, 3);
disp(b);
disp(a);
