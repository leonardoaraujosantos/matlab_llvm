% Tier-1 (Signal Processing Toolbox roadmap §2.4): polynomial helpers.
% Exercises roots / poly / polyder / polyint on small polynomials whose
% answers are exact in IEEE-754. Print scalar reductions to keep the
% golden small + lane-stable.

% (x - 1)(x - 2) = x^2 - 3x + 2.
p1 = [1 -3 2];
r1 = roots(p1);
disp(real(r1));         % [2; 1] (Durand-Kerner ordering — sum/diff stable)

% poly(roots) round-trip: should recover p1 (up to leading-coefficient
% normalisation, which is 1 here).
p1_back = poly(r1);
disp(p1_back);          % [1 -3 2]

% polyder of x^3 + 2x^2 - x + 5 is 3x^2 + 4x - 1.
p2 = [1 2 -1 5];
disp(polyder(p2));      % [3 4 -1]

% polyint with default k = 0: integral of [1 0 0] (= x^2) is x^3 / 3.
disp(polyint([1 0 0])); % [0.333333 0 0 0]

% polyint with explicit constant k.
disp(polyint([2 0], 7));  % integral(2x) = x^2 + 7 -> [1 0 7]

% Sum of coefficients of poly(roots(p)) == sum(p) for any p with real
% roots — equivalent to evaluating both at x = 1.
disp(sum(poly([1 2 3 4])));     % p(1) for poly with roots 1,2,3,4: (1-1)(1-2)(1-3)(1-4) = 0
disp(sum(poly([0.5 -0.5 2])));  % (1-0.5)(1+0.5)(1-2) = -0.75
