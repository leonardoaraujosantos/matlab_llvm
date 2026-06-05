% Regression for #235: vector/array number-theory builtins.
%   primes(n)        -> row vector of all primes <= n
%   isprime([...])   -> element-wise logical (0/1), shape-preserving
%   factorial([...]) -> element-wise, shape-preserving
% The scalar forms isprime(x) / factorial(x) (shipped earlier, #229/#231)
% must still route to the scalar runtime path and return a scalar.
% Printed via fprintf %.0f (<=4 values/line) so output is identical across
% all four execute backends, mirroring regress_isprime_nchoosek.
p = primes(20);
fprintf('%.0f %.0f %.0f %.0f\n', numel(p), sum(p), p(1), p(8));
ip = isprime([4 5 6 7]);
fprintf('%.0f %.0f %.0f %.0f\n', ip(1), ip(2), ip(3), ip(4));
fa = factorial([0 2 4 5]);
fprintf('%.0f %.0f %.0f %.0f\n', fa(1), fa(2), fa(3), fa(4));
fprintf('%.0f %.0f\n', isprime(13), factorial(6));
