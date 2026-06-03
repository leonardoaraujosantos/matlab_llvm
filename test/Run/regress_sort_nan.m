% Regression: sort() must place NaN at the end, in both ascending and
% descending order (MATLAB). Previously the comparator used < / > (false for
% any NaN), leaving NaN unsorted in place. NaN is produced via rem(_,0) and
% detected backend-independently via v==v being false.
x = [3 rem(5,0) 1];        % [3 NaN 1]
s = sort(x);               % ascending -> [1 3 NaN]
fprintf('asc: %.0f %.0f\n', s(1), s(2));
v = s(3); if v == v; disp(1); else; disp(0); end   % NaN at end -> 0
d = sort(x, 'descend');    % descending -> [3 1 NaN]  (NaN still last)
fprintf('desc: %.0f %.0f\n', d(1), d(2));
w = d(3); if w == w; disp(1); else; disp(0); end   % NaN at end -> 0
n = sort([rem(1,0) 2 rem(1,0) 1]);   % [1 2 NaN NaN]
fprintf('multi: %.0f %.0f\n', n(1), n(2));
a = n(3); b = n(4);
if a == a; disp(1); else; disp(0); end   % 0
if b == b; disp(1); else; disp(0); end   % 0
