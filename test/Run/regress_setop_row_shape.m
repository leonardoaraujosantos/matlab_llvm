% Regression: intersect/setdiff/union must preserve input orientation like
% MATLAB. The result is a row vector only when BOTH inputs are row vectors;
% otherwise a column vector. Previously these always returned a column.
a = [1 2 3 4];
b = [3 4 5 6];
i = intersect(a, b);
fprintf('inter: %.0fx%.0f vals=%.0f %.0f\n', size(i,1), size(i,2), i(1), i(2));
d = setdiff(a, b);
fprintf('diff:  %.0fx%.0f vals=%.0f %.0f\n', size(d,1), size(d,2), d(1), d(2));
u = union(a, b);
fprintf('union: %.0fx%.0f\n', size(u,1), size(u,2));
% Mixed / column inputs stay columns.
ac = [1; 2; 3; 4];
bc = [3; 4; 5; 6];
ic = intersect(ac, bc);
fprintf('inter_col: %.0fx%.0f\n', size(ic,1), size(ic,2));
um = union(ac, b);
fprintf('union_mixed: %.0fx%.0f\n', size(um,1), size(um,2));
