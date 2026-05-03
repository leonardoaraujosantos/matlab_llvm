% Robertson reaction kinetics — the canonical stiff test problem from
% numerical-analysis textbooks. Three-component reaction with widely-
% separated time scales (rate constants 0.04, 1e4, 3e7).
%
%   y1' = -0.04*y1 + 1e4*y2*y3
%   y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
%   y3' =                        3e7*y2^2
%
% Conservation: y1 + y2 + y3 = constant = 1.

f = @(t,y) [(0 - 0.04*y(1) + 1e4*y(2)*y(3));
            (0.04*y(1) - 1e4*y(2)*y(3) - 3e7*y(2)*y(2));
            (3e7*y(2)*y(2))];
y0 = [1; 0; 0];

[t, y, stats] = ode23s(f, [0 1], y0);

% Vector ode23s should integrate Robertson over [0, 1] in well under
% 100 steps (FD Jacobian + LU + back-solves per step).
if stats.nsteps < 100; disp(1); else; disp(0); end

% Conservation: y1 + y2 + y3 should still be 1 at t = 1.
total = y(end, 1) + y(end, 2) + y(end, 3);
if abs(total - 1) < 1e-6; disp(1); else; disp(0); end

% y2 is a fast transient — it should be tiny (steady-state
% concentration is around 1e-5).
if y(end, 2) < 1e-3; disp(1); else; disp(0); end
