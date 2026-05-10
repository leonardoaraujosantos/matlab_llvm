% ss operator overloads — §3.1 sibling math.
%
%   ss + ss / ss - ss → parallel: A is block-diagonal, B is stacked,
%                       C is concatenated, D is summed (or differenced).
%   ss * ss          → series cascade: a*b means u → b → a → y; the
%                       state is [x_a; x_b] with A = [A_a, B_a*C_b;
%                       0, A_b], etc. (see cst_class_ss.m).
%   -ss              → negate output: C → -C, D → -D.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_model_objects.

A1 = [-1 0; 0 -2];
B1 = [1; 1];
C1 = [1 0];
D1 = 0;
sys1 = ss(A1, B1, C1, D1);

A2 = [-3];
B2 = [1];
C2 = [2];
D2 = 0;
sys2 = ss(A2, B2, C2, D2);

% --- plus: A is block_diag([-1,0;0,-2], [-3]) = 3×3.
P = sys1 + sys2;
disp(P.A);
disp(P.B);
disp(P.C);
disp(P.D);

% --- minus: same A/B but C has b.C negated.
N = sys1 - sys2;
disp(N.C);
disp(N.D);

% --- uminus: C → -C.
M = -sys1;
disp(M.C);

% --- mtimes: series cascade (sys2 first, then sys1).
S = sys1 * sys2;
disp(S.A);
disp(S.B);
disp(S.C);
disp(S.D);
