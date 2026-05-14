% linprog — Optimization Toolbox Tier-1.5.  Linear programming via a
% dense 2-phase tableau simplex.  See docs/optim_toolbox_roadmap.md.
%
%   x = linprog(f, A, b)                    % 3-arg, default lb = 0
%   x = linprog(f, A, b, Aeq, beq, lb, ub)  % full 7-arg form
%
% Tier-1 contract: lower bounds default to 0 when lb is absent.
% Multi-component checks use a summed absolute error against one
% threshold (the LLVM lane does not lower `&&` as a value).
%
% Test LP (a small "diet"-style problem):
%   minimize  -x1 - 2*x2
%   s.t.       x1 +   x2 <= 4
%              x1 + 3*x2 <= 6
%              x >= 0
% Vertex enumeration: (4,0) → -4, (0,2) → -4, (3,1) → -5.
% Optimum is x = [3; 1], objective -5.

f = [-1; -2];
A = [1, 1; 1, 3];
b = [4; 6];

% --- 1. 3-argument form (default lower bound 0) -------------------
x1 = linprog(f, A, b);
e1 = abs(x1(1) - 3) + abs(x1(2) - 1);
if e1 < 2e-6; disp(1); else; disp(0); end

% --- 2. Full 7-argument form with explicit bounds -----------------
lb = [0; 0];
ub = [10; 10];
x2 = linprog(f, A, b, [], [], lb, ub);
e2 = abs(x2(1) - 3) + abs(x2(2) - 1);
if e2 < 2e-6; disp(1); else; disp(0); end

% --- 3. Objective value at the optimum is -5 ---------------------
obj = f(1)*x1(1) + f(2)*x1(2);
if abs(obj + 5) < 1e-6; disp(1); else; disp(0); end

% --- 4. Equality constraint: x1 + x2 = 3 with same objective ------
%   minimize -x1 - 2*x2  s.t.  x1 + x2 = 3 (and the <= rows),  x >= 0
%   Objective is -3 - x2, so push x2 up; the row x1 + 3*x2 <= 6 with
%   x1 = 3 - x2 gives 3 + 2*x2 <= 6 → x2 <= 1.5.  → x = [1.5; 1.5].
Aeq = [1, 1];
beq = 3;
x3 = linprog(f, A, b, Aeq, beq, lb, ub);
e3 = abs(x3(1) - 1.5) + abs(x3(2) - 1.5);
if e3 < 2e-6; disp(1); else; disp(0); end

% --- 5. Upper bound becomes binding ------------------------------
%   same objective, but cap x2 <= 0.5  → x = [3.5; 0.5], obj -4.5.
ub2 = [10; 0.5];
x4 = linprog(f, A, b, [], [], lb, ub2);
e4 = abs(x4(1) - 3.5) + abs(x4(2) - 0.5);
if e4 < 2e-6; disp(1); else; disp(0); end
