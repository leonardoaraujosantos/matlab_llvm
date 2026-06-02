% regress_end_single_subscript.m — regression test for `end` in
% single-subscript indexing (#136). Before the fix, `end` always
% resolved to size(base, dim) where dim = the 1-based subscript
% position, so a single subscript used dim=1 (= number of rows).
% For a row vector that is 1, not numel, so v(end) returned the
% first element instead of the last. The lowering now passes a
% sentinel dim 0 for single-subscript indexing and matlab_end_of_dim
% maps dim 0 to numel(base).

% --- row vector: end == numel, not size(,1)=1 --------------------
v = [10 20 30 40];
disp(v(end));        % 40
disp(v(end-1));      % 30
disp(v(end-2));      % 20

% --- column vector --------------------------------------------------
c = [1; 2; 3];
disp(c(end));        % 3
disp(c(end-1));      % 2

% --- matrix linear single-subscript: end == numel ------------------
M = [1 2 3; 4 5 6];  % 2x3, numel 6
disp(M(end));        % 6  (last linear element)

% --- 2-D subscript still per-dimension ------------------------------
disp(M(end, end));   % (2,3) = 6
disp(M(1, end));     % (1,3) = 3

% --- end inside arithmetic on a row vector --------------------------
w = [5 6 7 8 9];
disp(w(end) + w(1)); % 9 + 5 = 14
