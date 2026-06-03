% Regression: concatenation must drop an empty [] operand (MATLAB), not return
% empty. Previously `[[] X]` / `[X []]` returned []. This is the grow-from-[]
% idiom. (#204)
x = [];
x = [x 1];
x = [x 2];
x = [x 3];
fprintf('grow: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(x,1), size(x,2), x(1), x(2), x(3));
a = [1 []];                 % trailing empty dropped -> [1]
fprintf('trail: %.0fx%.0f val=%.0f\n', size(a,1), size(a,2), a(1));
b = [[] 5];                 % leading empty dropped -> [5]
fprintf('lead: %.0f\n', b(1));
c = [];
for k = 1:3; c = [c; k*10]; end   % column grow -> [10;20;30]
fprintf('colgrow: %.0fx%.0f vals=%.0f %.0f %.0f\n', size(c,1), size(c,2), c(1), c(2), c(3));
d = [[]; [7 8]];            % leading empty row dropped -> [7 8]
fprintf('vrow: %.0f %.0f\n', d(1), d(2));
e = [1 2 3 4];              % no empties, unchanged
fprintf('plain: %.0f %.0f %.0f %.0f\n', e(1), e(2), e(3), e(4));
