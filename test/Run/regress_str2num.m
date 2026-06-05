% Regression for #235: str2num parses a numeric-literal / array string into
% a matrix (scalar / row / 2-D), with whitespace or comma separators and an
% optional surrounding [ ]. Invalid input yields an empty matrix (numel 0),
% matching MATLAB. Printed via fprintf %.0f so output is byte-identical
% across all four execute backends.
a = str2num('3.14');
fprintf('%.2f %.0f\n', a, numel(a));
v = str2num('[1 2 3 4]');
fprintf('%.0f %.0f %.0f\n', numel(v), sum(v), v(3));
w = str2num('5 6 7');
fprintf('%.0f %.0f\n', numel(w), sum(w));
M = str2num('[1 2; 3 4]');
fprintf('%.0f %.0f %.0f %.0f\n', M(1,1), M(1,2), M(2,1), M(2,2));
c = str2num('1,2,3');
fprintf('%.0f %.0f\n', numel(c), sum(c));
e = str2num('abc');
fprintf('%.0f\n', numel(e));
