% coneprog — Optimization Toolbox Tier-3.  Second-order cone
% programming: minimise f'x subject to ||Asc*x + bsc|| <= dsc'x +
% gamma, handled as a single nonlinear inequality through the
% augmented-Lagrangian core.  See docs/optim_toolbox_roadmap.md §4.
%
%   x = coneprog(f, Asc, bsc, dsc, gamma, A, b, Aeq, beq, lb, ub)
%
% Tier-3 supports a single second-order cone constraint.

% --- 1. maximise x1 + x2 on the unit disk ------------------------
%   minimise -x1 - x2  s.t.  ||x|| <= 1  (Asc = I, bsc = 0, dsc = 0,
%   gamma = 1).  The optimum is x = [1; 1] / sqrt(2) ~ [0.7071;0.7071].
f = [-1; -1];
Asc = [1, 0; 0, 1];
bsc = [0; 0];
dsc = [0; 0];
x = coneprog(f, Asc, bsc, dsc, 1);
e1 = abs(x(1) - 0.70710678) + abs(x(2) - 0.70710678);
if e1 < 1e-3; disp(1); else; disp(0); end

% --- 2. the cone constraint is (nearly) active -------------------
nrm = sqrt(x(1)*x(1) + x(2)*x(2));
if abs(nrm - 1) < 1e-3; disp(1); else; disp(0); end

% --- 3. minimise -x1 alone on the unit disk (11-arg form) --------
%   The optimum slides to x = [1; 0].
y = coneprog([-1; 0], Asc, bsc, dsc, 1, [], [], [], [], [], []);
e3 = abs(y(1) - 1) + abs(y(2) - 0);
if e3 < 1e-3; disp(1); else; disp(0); end

% --- 4. shifted / scaled cone: ||x|| <= 0.5 ----------------------
%   gamma = 0.5 shrinks the feasible ball; maximising x1 + x2 gives
%   x = [0.5; 0.5] / sqrt(2) ~ [0.35355; 0.35355].
z = coneprog(f, Asc, bsc, dsc, 0.5);
e4 = abs(z(1) - 0.35355339) + abs(z(2) - 0.35355339);
if e4 < 1e-3; disp(1); else; disp(0); end
