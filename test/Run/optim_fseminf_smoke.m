% fseminf — Optimization Toolbox Tier-3.  Semi-infinite programming:
% minimise fun(x) subject to phi(x, w) <= 0 for every w in [0, 1].
% The solver runs an outer-approximation loop — minimise over the
% currently sampled w-points, then add the most-violating w from a
% fine grid — until the worst violation is within tolerance.  See
% docs/optim_toolbox_roadmap.md §4.
%
%   x = fseminf(@fun, x0, @seminfcon)
%
% Tier-3 supports a single semi-infinite constraint whose handle has
% the per-point ABI seminfcon(x, w) -> scalar.

% --- 1. quadratic objective, linear-in-w constraint --------------
%   minimise (x1-1)^2 + (x2-1)^2
%   s.t.     x1*w + x2*(1-w) - 0.5 <= 0   for all w in [0,1]
%   The constraint is linear in w, so its worst case is at an
%   endpoint: it reduces to x1 <= 0.5 AND x2 <= 0.5.  The objective
%   then pulls the solution to the corner x = [0.5; 0.5].
fun = @(x) (x(1) - 1)*(x(1) - 1) + (x(2) - 1)*(x(2) - 1);
seminfcon = @(x, w) x(1)*w + x(2)*(1 - w) - 0.5;
x = fseminf(fun, [0; 0], seminfcon);
e1 = abs(x(1) - 0.5) + abs(x(2) - 0.5);
if e1 < 1e-2; disp(1); else; disp(0); end

% --- 2. the semi-infinite constraint is satisfied across w -------
%   Check phi(x, w) <= tol at a few sample points.
v0  = seminfcon(x, 0.0);
v05 = seminfcon(x, 0.5);
v1  = seminfcon(x, 1.0);
worst = v0;
if v05 > worst; worst = v05; end
if v1  > worst; worst = v1;  end
if worst < 1e-3; disp(1); else; disp(0); end

% --- 3. the constraint is active (the solver pushed up to it) ----
if worst > -1e-2; disp(1); else; disp(0); end

% --- 4. bounded fseminf (5-arg form) -----------------------------
%   Same problem but cap x1 <= 0.3; the corner moves to [0.3; 0.5].
xb = fseminf(fun, [0; 0], seminfcon, [-10; -10], [0.3; 10]);
e4 = abs(xb(1) - 0.3) + abs(xb(2) - 0.5);
if e4 < 2e-2; disp(1); else; disp(0); end
