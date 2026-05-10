% frd operator overloads — §3.1 sibling math.
%
% Element-wise on ResponseData, sharing the operand's Frequency
% vector. Mirrors frequency-domain semantics:
%   frd_a + frd_b   parallel sum at each ω
%   frd_a - frd_b   parallel difference
%   frd_a * frd_b   series cascade — H_ab(jω) = H_a(jω) · H_b(jω)
%                   (pointwise, not matrix multiply)
%   -frd_a          response negation
%
% Operands must share the same Frequency grid; true grid-mismatch
% interpolation is a follow-on.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_model_objects.

H1 = frd([1.0; 0.5; 0.25], [1; 10; 100]);
H2 = frd([2.0; 1.0; 0.5],  [1; 10; 100]);

% --- plus
P = H1 + H2;
disp(P.ResponseData);
disp(P.Frequency);

% --- minus
D = H1 - H2;
disp(D.ResponseData);

% --- mtimes (element-wise on response)
T = H1 * H2;
disp(T.ResponseData);

% --- uminus
N = -H1;
disp(N.ResponseData);
