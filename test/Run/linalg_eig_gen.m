% Generalised eigenvalue problem — `eig(A, B)` over the pencil
% A − λB. Tier-1.1 closure: small wrapper over the already-shipped
% 4-return qz(A, B). Returns a real column matrix when the spectrum
% is purely real; flips to a complex matrix when any conjugate pair
% appears (matches matlab_eig's polymorphic return).

% --- B = identity recovers eig(A).
A = [4 -2; 1 1];
B = eye(2);
disp(eig(A, B));

% --- Non-identity B: solve det(A - λ B) = 0 on a 2×2 pencil.
A2 = [1 2; 3 4];
B2 = [1 0; 0 2];
% (1-λ)(4-2λ) - 6 = 2λ² - 6λ - 2 = 0 → λ ≈ -0.303, 3.303
disp(eig(A2, B2));

% --- 3×3 diagonal pencil: eigenvalues = diag(A) ./ diag(B).
A3 = diag([-1 -2 -3]);
B3 = diag([1 2 1]);
disp(eig(A3, B3));

% --- Complex pencil: rotation pencil with B = I gives ±i.
A4 = [0 -1; 1 0];
B4 = eye(2);
disp(eig(A4, B4));
