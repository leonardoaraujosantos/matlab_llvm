% Regression: string elements of a cell array must retrieve as strings, not
% numeric char codes. Previously disp(c{i}) printed char codes (#206).
c = {"a", "bee", "cee"};
disp(c{1});                 % a
disp(c{2});                 % bee
t = c{3}; disp(t);          % cee  (string propagates through assignment)
n = {"x", 42};              % mixed cell: numeric element unaffected
fprintf('num=%.0f\n', n{2});
m = {[1 2 3], [4 5 6]};     % matrix cell unaffected
v = m{1};
fprintf('mat=%.0f %.0f %.0f\n', v(1), v(2), v(3));
