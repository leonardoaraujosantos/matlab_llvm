% regress_max_min_two_scalars.m — regression test for max/min with two
% scalar arguments, e.g. `max(3, 5)` (#153). Before the fix the pde_table
% had no two-scalar ("ff") shape for max/min — only the reduction `max(v)`,
% the element-wise `max(A,B)`, and `max(A,[],dim)` — so `max(3,5)` failed
% with "unsupported call shape". The two-scalar form now returns a 1x1
% matrix (matching the frontend's ptr typing for max/min, so the result
% flows through `+`/`max(_,s)` etc.); matrix-scalar broadcast forms
% `max(A,s)` / `max(s,A)` were added alongside.

% --- two scalars ---------------------------------------------------
disp(max(3, 5));        % 5
disp(min(2, 9));        % 2
disp(max(-5, -1));      % -1
disp(min(-5, -1));      % -5

% --- result flows into arithmetic ----------------------------------
disp(max(3, 8) + 0);    % 8
a = 4; b = 7; c = 5;
disp(max(a, b) + c);    % 12

% --- nested (inner 1x1 feeds outer max(_, scalar)) -----------------
disp(max(max(1, 2), 3));    % 3
disp(min(min(9, 4), 7));    % 4

% --- matrix-scalar broadcast (both operand orders) -----------------
disp(sum(max([1 5 3], 3)));  % [3 5 3] -> 11
disp(sum(min(3, [2 4 1])));  % [2 3 1] -> 6

% --- existing forms still work -------------------------------------
disp(max([3 7 2]));          % reduction -> 7
disp(sum(max([1 5 2], [3 4 4])));  % elementwise [3 5 4] -> 12
