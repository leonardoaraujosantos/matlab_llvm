% intlinprog — Optimization Toolbox Tier-3.  A 3x3 assignment problem
% solved as a binary program.  The assignment polytope is totally
% unimodular, so the LP relaxation at the branch-and-bound root is
% already integral.  See docs/optim_toolbox_roadmap.md §4.
%
%   minimise  sum_ij C_ij x_ij
%   s.t.      each row sums to 1, each column sums to 1, x in {0,1}
%
% Cost matrix C = [4 1 3; 2 0 5; 3 2 2]; variables x_11..x_33 in
% row-major order.  Enumerating the 6 permutations, the cheapest
% assignment is 1->2, 2->1, 3->3 with cost 1 + 2 + 2 = 5, i.e.
% x = [0 1 0; 1 0 0; 0 0 1] flattened to [0 1 0 1 0 0 0 0 1].

f = [4; 1; 3; 2; 0; 5; 3; 2; 2];
intcon = [1; 2; 3; 4; 5; 6; 7; 8; 9];
% Row sums = 1.
Aeq = [1, 1, 1, 0, 0, 0, 0, 0, 0; ...
       0, 0, 0, 1, 1, 1, 0, 0, 0; ...
       0, 0, 0, 0, 0, 0, 1, 1, 1; ...
       1, 0, 0, 1, 0, 0, 1, 0, 0; ...
       0, 1, 0, 0, 1, 0, 0, 1, 0; ...
       0, 0, 1, 0, 0, 1, 0, 0, 1];
beq = [1; 1; 1; 1; 1; 1];
lb = [0; 0; 0; 0; 0; 0; 0; 0; 0];
ub = [1; 1; 1; 1; 1; 1; 1; 1; 1];
x = intlinprog(f, intcon, [], [], Aeq, beq, lb, ub);

% --- 1. the expected assignment was found ------------------------
expected = [0; 1; 0; 1; 0; 0; 0; 0; 1];
e1 = 0;
for k = 1:9
    e1 = e1 + abs(x(k) - expected(k));
end
if e1 < 1e-6; disp(1); else; disp(0); end

% --- 2. total cost is 5 ------------------------------------------
cost = 0;
for k = 1:9
    cost = cost + f(k)*x(k);
end
if abs(cost - 5) < 1e-6; disp(1); else; disp(0); end

% --- 3. every variable is 0 or 1 ---------------------------------
fr = 0;
for k = 1:9
    fr = fr + abs(x(k) - round(x(k)));
end
if fr < 1e-6; disp(1); else; disp(0); end
