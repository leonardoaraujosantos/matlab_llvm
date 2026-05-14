% fzero — Optimization Toolbox Tier-1.1.  Brent's method for scalar
% root finding.  See docs/optim_toolbox_roadmap.md.
%
% Exercises both operand shapes the lowering recognises:
%   x = fzero(@fn, x0)     — scalar guess, bracket auto-expanded
%   x = fzero(@fn, [a b])  — sign-change bracket supplied
%
% Reference values:
%   cos(x) - x = 0    → Dottie number 0.739085133215161
%   sin(x) = 0 near 3 → π
%   x^3 - x - 2 = 0   → 1.521379706804568

% --- 1. cos(x) - x = 0, scalar-guess form -------------------------
f1 = @(x) cos(x) - x;
r1 = fzero(f1, 0.5);
if abs(r1 - 0.7390851332151607) < 1e-9; disp(1); else; disp(0); end

% --- 2. sin(x) = 0, bracket form picks π not 0 --------------------
f2 = @(x) sin(x);
r2 = fzero(f2, [3, 4]);
if abs(r2 - 3.141592653589793) < 1e-9; disp(1); else; disp(0); end

% --- 3. x^3 - x - 2 = 0, scalar guess away from root --------------
f3 = @(x) x*x*x - x - 2;
r3 = fzero(f3, 0);
if abs(r3 - 1.521379706804568) < 1e-9; disp(1); else; disp(0); end

% --- 4. Bracket form against the same cubic -----------------------
r4 = fzero(f3, [1, 2]);
if abs(r4 - 1.521379706804568) < 1e-9; disp(1); else; disp(0); end

% --- 5. Initial guess exactly at the root -------------------------
r5 = fzero(f1, 0.7390851332151607);
if abs(r5 - 0.7390851332151607) < 1e-12; disp(1); else; disp(0); end
