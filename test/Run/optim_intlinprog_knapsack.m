% intlinprog — Optimization Toolbox Tier-3.  Mixed-integer linear
% programming by depth-first branch-and-bound over the dense 2-phase
% simplex (linprog_core).  See docs/optim_toolbox_roadmap.md §4.
%
%   x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)
%
% Multi-component checks use a summed absolute error against one
% threshold (the LLVM lane does not lower `&&` as a value).

% --- 1. 0/1 knapsack ---------------------------------------------
%   maximise  8*x1 + 11*x2 + 6*x3 + 4*x4   (so minimise the negation)
%   s.t.      5*x1 + 7*x2 + 4*x3 + 3*x4 <= 10,   x in {0,1}
%   Enumerating feasible subsets, the best is {item2, item4}: value
%   15, weight 10.  Optimum x = [0; 1; 0; 1].
f = [-8; -11; -6; -4];
intcon = [1; 2; 3; 4];
A = [5, 7, 4, 3];
b = 10;
lb = [0; 0; 0; 0];
ub = [1; 1; 1; 1];
x = intlinprog(f, intcon, A, b, [], [], lb, ub);
e1 = abs(x(1) - 0) + abs(x(2) - 1) + abs(x(3) - 0) + abs(x(4) - 1);
if e1 < 1e-6; disp(1); else; disp(0); end

% --- 2. objective value at the optimum is -15 --------------------
obj = f(1)*x(1) + f(2)*x(2) + f(3)*x(3) + f(4)*x(4);
if abs(obj + 15) < 1e-6; disp(1); else; disp(0); end

% --- 3. the knapsack capacity is respected -----------------------
wt = 5*x(1) + 7*x(2) + 4*x(3) + 3*x(4);
if wt <= 10 + 1e-6; disp(1); else; disp(0); end

% --- 4. the returned variables are integers ----------------------
fr = abs(x(1) - round(x(1))) + abs(x(2) - round(x(2))) + ...
     abs(x(3) - round(x(3))) + abs(x(4) - round(x(4)));
if fr < 1e-6; disp(1); else; disp(0); end

% --- 5. tighter capacity forces a different pick -----------------
%   capacity 7: feasible subsets are {2}=11, {1,4}=12, {3,4}=10,
%   {1}=8, ...  best is {1,4} with value 12, weight 8 > 7 — no.
%   Recheck: {1,4} weight 5+3=8 > 7.  Feasible within 7:
%   {2} w7 v11, {3,4} w7 v10, {1,3} w9 no, {1} w5 v8, {4} w3 v4.
%   Best is {2}: x = [0; 1; 0; 0], value 11.
x2 = intlinprog(f, intcon, A, 7, [], [], lb, ub);
e5 = abs(x2(1)) + abs(x2(2) - 1) + abs(x2(3)) + abs(x2(4));
if e5 < 1e-6; disp(1); else; disp(0); end
