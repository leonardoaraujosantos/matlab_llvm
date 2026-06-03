% regress_sort_direction.m — regression test for sort(x, 'ascend'|'descend')
% (#167). Before the fix only the 1-arg sort(x) was wired; the 2-arg
% direction form failed with "unsupported call shape". The direction
% const_char is now materialised to a matlab_string* and routed to
% matlab_sort_dir, which sorts ascending or descending. Element-wise scalar
% disps so the order is checked and the output is backend-independent.

% --- descending -----------------------------------------------------
d = sort([3 1 9 2 5], 'descend');   % [9 5 3 2 1]
disp(d(1));    % 9
disp(d(2));    % 5
disp(d(5));    % 1

% --- ascending (explicit) ------------------------------------------
a = sort([3 1 9 2 5], 'ascend');    % [1 2 3 5 9]
disp(a(1));    % 1
disp(a(5));    % 9

% --- default (still ascending) -------------------------------------
b = sort([2 1 3]);                  % [1 2 3]
disp(b(1));    % 1
disp(b(3));    % 3

% --- descending of an already-sorted vector ------------------------
e = sort([1 2 3 4], 'descend');     % [4 3 2 1]
disp(e(1));    % 4
disp(e(4));    % 1
