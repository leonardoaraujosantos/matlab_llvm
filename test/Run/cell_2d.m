% Phase 1.3 — 2-D cells and cell concatenation. Three idioms:
%   1. 2-D cell literal `{a, b; c, d}` -> cell_new_2d + per-cell set.
%   2. C{r, k} indexing reads / writes to the (r, k) slot.
%   3. `[A, B]` / `[A; B]` where A/B are cells -> cell_concat_row / _col.

% 2-D cell literal.
C = {1, 2, 3; 4, 5, 6};
disp(C{1,1});
disp(C{1,2});
disp(C{2,3});

% Shape of a 2-D cell.
disp(size(C, 1));
disp(size(C, 2));

% Cell concatenation: row.
A = {10, 20};
B = {30, 40};
R = [A, B];
disp(size(R, 1));
disp(size(R, 2));
disp(R{1,1});
disp(R{1,4});

% Cell concatenation: column.
P = {100, 200};
Q = {300, 400};
S = [P; Q];
disp(size(S, 1));
disp(size(S, 2));
disp(S{1,2});
disp(S{2,1});

% 2-D write then read on a literal-allocated cell.
T = {0, 0; 0, 0};
T{1,2} = 99;
T{2,1} = 7;
T{2,2} = 8;
disp(T{1,2});
disp(T{2,1});
disp(T{2,2});

% Mixed concat (cell with a different cell shape).
U = {1, 2; 3, 4};
V = {5, 6; 7, 8};
W = [U; V];                      % 4x2 cell
disp(size(W, 1));
disp(size(W, 2));
disp(W{3,1});
disp(W{4,2});
