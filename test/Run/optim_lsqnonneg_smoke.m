% lsqnonneg — Optimization Toolbox Tier-1.6.  Non-negative least
% squares via the Lawson-Hanson active-set algorithm.  See
% docs/optim_toolbox_roadmap.md.
%
%   x = lsqnonneg(C, d)   % minimise ||C*x - d||^2 subject to x >= 0
%
% Multi-component checks use a summed absolute error against one
% threshold (the LLVM lane does not lower `&&` as a value).

% --- 1. Constraint inactive: NNLS == ordinary least squares ------
%   C'C = [2 1; 1 2],  C'd = [5; 6]  → x = [4/3; 7/3], both positive.
C1 = [1, 0; 0, 1; 1, 1];
d1 = [1; 2; 4];
x1 = lsqnonneg(C1, d1);
e1 = abs(x1(1) - 1.333333333333333) + abs(x1(2) - 2.333333333333333);
if e1 < 1e-6; disp(1); else; disp(0); end

% --- 2. Constraint active on one variable ------------------------
%   Same C, d = [-1; 2; 1].  Unconstrained LS would give x1 = -1 < 0;
%   NNLS pins x1 = 0 and fits x2 alone → x2 = (2 + 1)/2 = 1.5.
d2 = [-1; 2; 1];
x2 = lsqnonneg(C1, d2);
e2 = abs(x2(1)) + abs(x2(2) - 1.5);
if e2 < 1e-6; disp(1); else; disp(0); end

% --- 3. All-negative target: solution pinned at the origin -------
C3 = [1; 2; 3];
d3 = [-1; -2; -3];
x3 = lsqnonneg(C3, d3);
if abs(x3(1)) < 1e-9; disp(1); else; disp(0); end

% --- 4. Exactly solvable non-negative system ---------------------
%   C = I(3), d = [0.5; 1.5; 2.5]  → x = d (all non-negative).
C4 = [1, 0, 0; 0, 1, 0; 0, 0, 1];
d4 = [0.5; 1.5; 2.5];
x4 = lsqnonneg(C4, d4);
e4 = abs(x4(1) - 0.5) + abs(x4(2) - 1.5) + abs(x4(3) - 2.5);
if e4 < 1e-9; disp(1); else; disp(0); end
