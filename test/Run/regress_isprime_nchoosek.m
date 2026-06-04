% isprime(n) -> logical; nchoosek(n,k) -> binomial coefficient. Both were
% previously unrecognised. Extends the scalar-math builtin set.
fprintf('%.0f %.0f %.0f %.0f\n', isprime(7), isprime(8), isprime(2), isprime(1));
fprintf('%.0f %.0f %.0f %.0f\n', nchoosek(5,2), nchoosek(10,0), nchoosek(6,3), nchoosek(4,4));
n = 13; fprintf('var %.0f\n', isprime(n));
