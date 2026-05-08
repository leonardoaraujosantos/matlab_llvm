% Polynomial helpers — roots / poly / polyder / polyint / residue.
%
% These power filter design (roots of A(z) gives the poles of the
% transfer function H(z) = B(z)/A(z); poly(roots) reconstructs the
% polynomial; residue does partial-fraction expansion for closed-form
% inverse Z-transforms).

% Roots of a polynomial p(x) = x^4 - 10x^3 + 35x^2 - 50x + 24
%   = (x-1)(x-2)(x-3)(x-4) → roots {1, 2, 3, 4}
% Durand-Kerner returns roots in solver-order; assert via the
% Vieta's-formulas symmetric functions instead.
p = [1 -10 35 -50 24];
r = roots(p);
disp('sum of roots:');
disp(sum(real(r)));    % 1+2+3+4 = 10
disp('product of roots:');
disp(prod(real(r)));   % 1*2*3*4 = 24

% Round-trip: poly(roots) should give back the polynomial.
p2 = poly(r);
fprintf('p2 length: %g\n', length(p2));

% Derivative: d/dx (x^4 - 10x^3 + 35x^2 - 50x + 24) = 4x^3 - 30x^2 + 70x - 50.
dp = polyder(p);
disp('derivative:');
disp(dp);

% Antiderivative with integration constant 0.
ip = polyint(p);
disp('integral:');
disp(ip);

% Partial-fraction expansion via residue: B(s)/A(s) = (s + 5) / (s^2 + 3s + 2)
%   = 4/(s + 1) - 3/(s + 2) (cover-up rule on simple poles)
b = [1 5];
a = [1 3 2];
[r, p, k] = residue(b, a);
disp('residues:');
disp(r);
disp('poles:');
disp(p);
disp('direct term k:');
disp(k);
