% regress_reverse_index.m — regression test for indexing a vector with a
% same-length index list (#165). Before the fix, matlab_slice1 (and the
% store path) treated any same-shape index as a logical mask, so a reorder
% like v([3 2 1]) or the reverse idiom v(end:-1:1) returned the elements in
% their original order. A logical mask only ever holds 0/1, so a same-shape
% index with a value outside {0,1} is now treated as an index list.
% Element-wise scalar disps so the output is order-sensitive AND
% backend-formatting-independent.

% --- reverse via descending range ----------------------------------
v = [10 20 30];
r = v(end:-1:1);     % [30 20 10]
disp(r(1));          % 30
disp(r(2));          % 20
disp(r(3));          % 10

% --- explicit reorder ----------------------------------------------
g = v([3 1 2]);      % [30 10 20]
disp(g(1));          % 30
disp(g(2));          % 10
disp(g(3));          % 20

% --- reorder on store ----------------------------------------------
w = [1 2 3];
w([3 2 1]) = [7 8 9];   % w(3)=7, w(2)=8, w(1)=9 -> [9 8 7]
disp(w(1));          % 9
disp(w(2));          % 8
disp(w(3));          % 7

% --- logical mask (0/1) still works as a mask ----------------------
m = [10 20 30 40];
disp(sum(m(m > 15)));   % 20+30+40 = 90
mm = [1 2 3 4];
mm(mm > 2) = 0;
disp(sum(mm));          % 1+2+0+0 = 3

% --- ascending / strided indexing unaffected -----------------------
p = [10 20 30 40 50];
q = p(1:2:5);        % [10 30 50]
disp(q(2));          % 30
