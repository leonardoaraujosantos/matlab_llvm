% fsolve — Optimization Toolbox Tier-1.9.  Scalar nonlinear equation
% solver: Newton's method with a finite-difference derivative and a
% bracket-expansion + Brent fallback when Newton stalls.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = fsolve(@fn, x0)   % solve fn(x) = 0
%
% Reference roots:
%   x^2 - 2 = 0          → sqrt(2) = 1.414213562373095
%   cos(x) - x = 0       → Dottie number 0.739085133215161
%   x^3 - 6x^2 + 11x - 6 → roots 1, 2, 3 (Newton picks the nearest)
%   exp(x) - 3 = 0       → log(3) = 1.098612288668110

% --- 1. sqrt(2) via x^2 - 2 --------------------------------------
f1 = @(x) x*x - 2;
r1 = fsolve(f1, 1);
if abs(r1 - 1.414213562373095) < 1e-9; disp(1); else; disp(0); end

% --- 2. Dottie number via cos(x) - x -----------------------------
f2 = @(x) cos(x) - x;
r2 = fsolve(f2, 0);
if abs(r2 - 0.739085133215161) < 1e-9; disp(1); else; disp(0); end

% --- 3. Cubic with three real roots: Newton converges to nearest -
f3 = @(x) x*x*x - 6*x*x + 11*x - 6;
r3 = fsolve(f3, 2.2);
if abs(r3 - 2) < 1e-9; disp(1); else; disp(0); end

% --- 4. exp(x) - 3 = 0 → log(3) ----------------------------------
f4 = @(x) exp(x) - 3;
r4 = fsolve(f4, 0);
if abs(r4 - 1.098612288668110) < 1e-9; disp(1); else; disp(0); end
