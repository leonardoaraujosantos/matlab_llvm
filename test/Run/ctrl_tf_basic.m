% tf model object — §3.1 with tf-vs-tf operator overloads.
%
% Constructor `tf(num, den)` + property access (`obj.Numerator`,
% `obj.Denominator`) + tf-vs-tf operator overloads:
%   G + H  parallel    (a.num·b.den + b.num·a.den) / (a.den·b.den)
%   G - H  parallel-with-sign-flip
%   G * H  series cascade  (conv numerators, conv denominators)
%   G / H  right-divide   (conv num·den_other, den·num_other)
%   -G     unary negate of numerator
%
% Scalar mixing (`s + 2`, `tf('s')` builder) needs Sema-level CST
% property type tracking and is the next slice. That blocker keeps
% emit-c / cpp / python / ts skipped today: each emits the obj as a
% struct value rather than a pointer, which breaks the
% matlab_obj_get_mat call signature.

G = tf([1 2], [1 3 5]);
H = tf([1 1], [1 1]);

% --- Property reads
disp(G.Numerator);
disp(G.Denominator);

% --- Series cascade.
P = G * H;
disp(P.Numerator);
disp(P.Denominator);

% --- Parallel sum.
S = G + H;
disp(S.Numerator);
disp(S.Denominator);

% --- Parallel difference.
D = G - H;
disp(D.Numerator);
disp(D.Denominator);

% --- Unary minus.
M = -G;
disp(M.Numerator);
disp(M.Denominator);

% --- Right-divide.
Q = G / H;
disp(Q.Numerator);
disp(Q.Denominator);
