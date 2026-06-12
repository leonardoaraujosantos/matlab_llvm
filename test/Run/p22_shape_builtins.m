% #191 P2.2 — shape-preserving elementwise builtins (sign/log10/tanh/bitshift)
% and reshape now carry a concrete type into downstream call args (were Any).
% These mirror the abs/floor pattern: ptr-in -> ptr-out, so no scalar
% box/unbox concern, and a scalar arg still lowers to an unboxed scalar.
v = [1 4 9 16];
a = chain(sign(v - 5));          % sign(vec) -> vec, fed into a user fn arg
fprintf('a %.0f\n', a);

L = log10([10 100 1000]);
fprintf('L %.4f %.4f %.4f\n', L(1), L(2), L(3));

T = tanh([0 100]);
fprintf('T %.0f %.0f\n', T(1), T(2));

R = reshape([1 2 3 4 5 6], 2, 3); % column-major -> [1 3 5; 2 4 6]
fprintf('R %.0f %.0f\n', R(1,1), R(2,3));

function s = chain(x)
  s = sum(x);                     % sum over the vector
end
