% zpk operator overloads — §3.1 sibling math.
%
%   zpk * zpk → series cascade: roots concatenate, gains multiply.
%   zpk / zpk → right-divide ≡ a * inv(b); inv swaps Z/P and 1/K.
%   -zpk     → negate gain; zeros and poles unchanged.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_model_objects.

G = zpk([-1; -3], [-2; -4], 5);
H = zpk([-5], [-6; -7], 2);

% --- mtimes: zeros = [-1; -3; -5], poles = [-2; -4; -6; -7], k = 10
P = G * H;
disp(P.Z);
disp(P.P);
disp(P.K);

% --- uminus: gain = -5, Z and P unchanged
M = -G;
disp(M.K);

% --- mrdivide: G / H = G * inv(H)
%     Z = [a.Z; b.P] = [-1; -3; -6; -7]
%     P = [a.P; b.Z] = [-2; -4; -5]
%     K = a.K / b.K = 5 / 2 = 2.5
Q = G / H;
disp(Q.Z);
disp(Q.P);
disp(Q.K);
