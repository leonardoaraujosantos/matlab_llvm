% regress_cell_matrix_elem.m — regression test for brace-reading a matrix
% element of a cell literal. Before the fix, `c{k}` defaulted to
% matlab_cell_get_f64 when Sema couldn't type the element (it can't carry
% per-element cell types), so a multi-element matrix slot was read as a
% scalar and printed 0. The lowering now records which cell-literal element
% indices are matrix/string-stored and forces matlab_cell_get_mat for a
% constant-index read of those slots (matlab_cell_get_mat is already
% kind-aware: it boxes scalars and converts strings).

% --- matrix element read back as a matrix, not 0 -------------------
c = {1, 5, [1 2 3]};
disp(c{3});          % 1 2 3   (was 0)

% --- leading matrix element ----------------------------------------
d = {[10 20 30], 9};
disp(d{1});          % 10 20 30

% --- matrix element usable downstream ------------------------------
e = {1, [4 5 6]};
disp(sum(e{2}));     % 15

% --- scalar elements are unaffected (still f64) --------------------
f = {7, 8, 9};
disp(f{3});          % 9
g = {10, 2};
disp(g{1} + g{2});   % 12

% --- 2-D matrix element --------------------------------------------
h = {99, [1 2; 3 4]};
disp(sum(sum(h{2}))); % 10
