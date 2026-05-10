% tf model object — §3.1 with tf-vs-tf operator overloads + scalar
% mixing (`G + 2`, `2 * G`).
%
% Constructor:
%   tf(num, den)       full polynomial form
%   tf(c)              scalar / vector boxed as `c / 1` (constant tf)
% Property access:
%   obj.Numerator, obj.Denominator
% Operators:
%   G + H / G - H      parallel sum / parallel-with-sign-flip
%   G * H              series cascade
%   G / H              right-divide (G * inv(H))
%   -G                 unary negate of numerator
% Scalar mixing: any operand may be a scalar / vector / raw matrix —
% the binary-op dispatch wraps it in a 1-arg `tf(c)` constructor
% before invoking the class method.
%
% `tf('s')` variable-builder is sugar for `tf([1 0], 1)`. We
% intercept the constructor call at the lowering site (char literals
% wouldn't survive the constructor body's slot-typed assignment), so
% `s = tf('s')` mints the same Laplace-variable transfer function as
% the explicit `s = tf([1 0], 1)` form. `tf('z')` lands the same
% nominal coefficients for discrete time; sample-time carry-through
% is a follow-on.
%
% LLVM-lane only — emit-c / cpp / python / ts skipped because each
% emits the obj as a struct value rather than a pointer, which
% doesn't match `matlab_obj_get_mat`'s `void*` first arg.

G = tf([1 2], [1 3 5]);
H = tf([1 1], [1 1]);

% --- Property reads.
disp(G.Numerator);
disp(G.Denominator);

% --- tf-vs-tf operators.
P = G * H;
disp(P.Numerator);
disp(P.Denominator);

S = G + H;
disp(S.Numerator);
disp(S.Denominator);

D = G - H;
disp(D.Numerator);
disp(D.Denominator);

M = -G;
disp(M.Numerator);
disp(M.Denominator);

Q = G / H;
disp(Q.Numerator);
disp(Q.Denominator);

% --- Scalar mixing.
% G + 1 = (s + 2)/(s²+3s+5) + 1 = (s² + 4s + 7)/(s² + 3s + 5)
A = G + 1;
disp(A.Numerator);
disp(A.Denominator);

% 5 * G = (5s + 10)/(s² + 3s + 5)
B = 5 * G;
disp(B.Numerator);
disp(B.Denominator);

% G - 1 = (s² - 1) ... wait: (s+2 - s²-3s-5)/(s²+3s+5) = (-s² -2s -3)/(s²+3s+5)
E = G - 1;
disp(E.Numerator);
disp(E.Denominator);

% 3 + G = (3s² + 10s + 17)/(s² + 3s + 5)
F = 3 + G;
disp(F.Numerator);
disp(F.Denominator);

% --- Polynomial-style composition with `sv = tf([1 0], 1)` as the
% Laplace variable. `(sv + 2) / (sv² + 3·sv + 5)` falls out
% compositionally and produces the same coefficients as the explicit
% `tf([1 2], [1 3 5])` constructor.
sv = tf([1 0], 1);
J = (sv + 2) / (sv * sv + 3 * sv + 5);
disp(J.Numerator);
disp(J.Denominator);

% --- `tf('s')` sugar — char-literal Laplace-variable form. Falls
% through the same constructor-call intercept as `tf([1 0], 1)` and
% should produce identical coefficients downstream.
sx = tf('s');
K = (sx + 2) / (sx * sx + 3 * sx + 5);
disp(K.Numerator);
disp(K.Denominator);
