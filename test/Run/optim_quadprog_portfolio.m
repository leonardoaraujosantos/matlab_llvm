% quadprog — Optimization Toolbox Tier-2.  Convex quadratic
% programming routed through the augmented-Lagrangian core with the
% analytic objective 1/2 x'Hx + f'x.  See docs/optim_toolbox_roadmap.md.
%
%   x = quadprog(H, f, A, b, Aeq, beq, lb, ub)
%
% Multi-component checks use a summed absolute error against one
% threshold (the LLVM lane does not lower `&&` as a value).

% --- 1. Minimum-variance portfolio (8-arg form) -------------------
%   minimise x'*Sigma*x  s.t.  x1 + x2 = 1,  x >= 0
%   with Sigma = [4 1; 1 2].  H = 2*Sigma; analytic optimum [0.25; 0.75].
H = [8, 2; 2, 4];
f = [0; 0];
Aeq = [1, 1];
beq = 1;
lb = [0; 0];
w = quadprog(H, f, [], [], Aeq, beq, lb, []);
e1 = abs(w(1) - 0.25) + abs(w(2) - 0.75);
if e1 < 1e-3; disp(1); else; disp(0); end

% --- 2. Portfolio weights sum to 1 --------------------------------
if abs(w(1) + w(2) - 1) < 1e-4; disp(1); else; disp(0); end

% --- 3. Equality-constrained QP (verifiable optimum) --------------
%   min x1^2 - 2x1 + x2^2 - 6x2  s.t.  x1 + x2 = 3
%   H = 2I, f = [-2;-6]; Lagrange solution x = [0.5; 2.5].
x2 = quadprog([2, 0; 0, 2], [-2; -6], [], [], [1, 1], 3, [], []);
e3 = abs(x2(1) - 0.5) + abs(x2(2) - 2.5);
if e3 < 1e-3; disp(1); else; disp(0); end

% --- 4. Bound-constrained QP (diagonal H decouples) ---------------
%   same H, f; bounds 0 <= x <= 1.  Unconstrained min is [1; 3],
%   so the bounds clamp the solution to [1; 1].
x3 = quadprog([2, 0; 0, 2], [-2; -6], [], [], [], [], [0; 0], [1; 1]);
e4 = abs(x3(1) - 1) + abs(x3(2) - 1);
if e4 < 1e-3; disp(1); else; disp(0); end

% --- 5. Inequality-constrained QP (4-arg form) --------------------
%   same H, f; x1 + x2 <= 3.  Unconstrained min [1;3] violates it,
%   so the optimum is the equality-constrained point [0.5; 2.5].
x4 = quadprog([2, 0; 0, 2], [-2; -6], [1, 1], 3);
e5 = abs(x4(1) - 0.5) + abs(x4(2) - 2.5);
if e5 < 1e-3; disp(1); else; disp(0); end
