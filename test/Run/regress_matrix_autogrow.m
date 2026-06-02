% regress_matrix_autogrow.m — #135: a numeric matrix/vector must auto-grow on
% an out-of-bounds indexed assignment (MATLAB semantics), matching the cell and
% struct-array stores. Before the fix, matlab_slice_store1_scalar skipped any
% index past the end (`if (lin >= m*n) continue`), so the write was silently
% dropped and the array kept its old size.
%
% NOTE: uses explicit out-of-bounds indices, NOT `v(end+1)` — single-subscript
% `end` is a separate bug (resolves to size(,1) not numel; see the companion
% issue), which would confound this auto-grow test on a row vector.

% Row vector grows along columns; the gap is zero-filled.
v = [1 2 3];
v(5) = 10;
disp(numel(v));     % 5
disp(v(5));         % 10
disp(v(4));         % 0  (gap)

% Column vector grows along rows.
c = [1; 2; 3];
c(5) = 7;
disp(size(c, 1));   % 5
disp(c(5));         % 7

% Empty grows into a row vector.
e = [];
e(3) = 9;
disp(numel(e));     % 3
disp(e(3));         % 9

% In-bounds assignment is unaffected (no spurious grow).
w = [1 2 3];
w(2) = 99;
disp(numel(w));     % 3
disp(w(2));         % 99
