% Scalar math builtins, all previously unrecognised: factorial, gcd, lcm,
% nextpow2, hypot, nthroot, log1p, expm1.
fprintf('%.0f %.0f %.0f %.0f %.0f\n', ...
  factorial(5), gcd(12,8), lcm(4,6), nextpow2(100), hypot(3,4));
fprintf('%.4f %.0f %.0f\n', nthroot(27,3), nthroot(-8,3), factorial(0));
fprintf('%.4f %.4f\n', log1p(0), expm1(0));
n = 6; fprintf('var %.0f\n', factorial(n));
