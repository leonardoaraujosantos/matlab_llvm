% fsolve — Optimization Toolbox Tier-2 (N-D form).  Solves a system
% of nonlinear equations F(x) = 0 by Levenberg-Marquardt on ||F(x)||^2.
% The scalar form (x0 a scalar) still routes to the Tier-1 Newton +
% Brent solver.  See docs/optim_toolbox_roadmap.md.
%
%   x = fsolve(@fun, x0)   % x0 vector → N-D ; x0 scalar → 1-D

% --- 1. 2x2 system: unit circle ∩ line x1 = x2 --------------------
%   F(x) = [x1^2 + x2^2 - 1; x1 - x2] = 0  → x1 = x2 = 1/sqrt(2).
F = @(x) [x(1)*x(1) + x(2)*x(2) - 1; x(1) - x(2)];
r = fsolve(F, [1; 1]);
e1 = abs(r(1) - 0.7071067812) + abs(r(2) - 0.7071067812);
if e1 < 1e-6; disp(1); else; disp(0); end

% --- 2. Residual at the solution is ~0 ----------------------------
res = F(r);
if res(1)*res(1) + res(2)*res(2) < 1e-10; disp(1); else; disp(0); end

% --- 3. 3x3 linear-ish system with a known root -------------------
%   F(x) = [x1 + x2 + x3 - 6; x1 - x2; x2 - x3] = 0 → x = [2; 2; 2].
G = @(x) [x(1) + x(2) + x(3) - 6; x(1) - x(2); x(2) - x(3)];
g = fsolve(G, [0; 0; 0]);
e3 = abs(g(1) - 2) + abs(g(2) - 2) + abs(g(3) - 2);
if e3 < 1e-6; disp(1); else; disp(0); end

% --- 4. Scalar form still works (Tier-1 path) ---------------------
s = fsolve(@(x) x*x - 2, 1);
if abs(s - 1.414213562373095) < 1e-9; disp(1); else; disp(0); end
