% Regression: unique() must preserve input orientation like MATLAB.
% A row-vector input yields a row vector; a column vector or matrix
% yields a column vector. Previously unique() always returned a column.
r = unique([3 1 1 2 3]);
fprintf('row: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(r,1), size(r,2), r(1), r(2), r(3));
c = unique([3; 1; 1; 2; 3]);
fprintf('col: %.0fx%.0f\n', size(c,1), size(c,2));
m = unique([3 1; 1 2]);
fprintf('mat: %.0fx%.0f\n', size(m,1), size(m,2));
