% fminbnd — Optimization Toolbox Tier-1.2.  1-D minimisation via
% Brent's method (golden-section + parabolic interpolation).  See
% docs/optim_toolbox_roadmap.md.
%
%   x = fminbnd(@fn, lo, hi)
%
% Reference minimisers:
%   (x-2)^2 + 1  on [0, 5]   → x = 2
%   sin(x)       on [3, 5]   → x = 3*pi/2 = 4.712388980384690
%   x^4 - 3x^3   on [0, 4]   → x = 9/4 = 2.25

% --- 1. Simple parabola, interior minimum --------------------------
f1 = @(x) (x - 2)*(x - 2) + 1;
m1 = fminbnd(f1, 0, 5);
if abs(m1 - 2) < 1e-6; disp(1); else; disp(0); end

% --- 2. sin(x): minimum at 3*pi/2 inside [3, 5] --------------------
f2 = @(x) sin(x);
m2 = fminbnd(f2, 3, 5);
if abs(m2 - 4.712388980384690) < 1e-6; disp(1); else; disp(0); end

% --- 3. Quartic x^4 - 3x^3, minimum at x = 9/4 --------------------
f3 = @(x) x*x*x*x - 3*x*x*x;
m3 = fminbnd(f3, 0, 4);
if abs(m3 - 2.25) < 1e-6; disp(1); else; disp(0); end

% --- 4. Reversed bracket is normalised internally -----------------
m4 = fminbnd(f1, 5, 0);
if abs(m4 - 2) < 1e-6; disp(1); else; disp(0); end
