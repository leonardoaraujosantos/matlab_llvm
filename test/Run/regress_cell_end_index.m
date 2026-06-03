% regress_cell_end_index.m — regression test for `end` inside cell
% brace-indexing (#175). Before the fix, `c{end}` left an unconverted
% matlab.end (the cell index arg was lowered without a SubscriptCtx base, and
% a cell `end` must resolve to matlab_cell_numel, not matlab_end_of_dim). The
% cell read now pushes a cell-numel sentinel and EndExpr routes it to
% matlab_cell_numel; the CellMatElems detection is extended so `c{end}` of a
% matrix element picks get_mat.

% --- end in cell brace, scalar elements ----------------------------
c = {10, 20, 30};
disp(c{end});       % 30
disp(c{end-1});     % 20
disp(c{end-2});     % 10

% --- longer cell ----------------------------------------------------
d = {1, 2, 3, 4, 5};
disp(d{end});       % 5

% --- end of a matrix element (CellMatElems via end) ----------------
e = {1, [7 8 9]};
f = e{end};         % [7 8 9]
disp(f(1));         % 7
disp(f(3));         % 9

% --- constant index still works ------------------------------------
disp(c{2});         % 20

% --- end arithmetic result used downstream -------------------------
last = c{end};
disp(last + 5);     % 35
