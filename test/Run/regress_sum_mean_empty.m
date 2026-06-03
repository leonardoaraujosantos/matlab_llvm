% Regression: sum([])==0 and mean([])==NaN (MATLAB), consistently across the
% AOT and shim backends. Previously the shims returned [] and AOT mean([])
% returned 0; NaN is detected via r==r being false. (#185)
fprintf('sum=%.0f\n', sum([]));           % 0
r = mean([]);
if r == r; disp(1); else; disp(0); end    % NaN -> 0
fprintf('sumv=%.0f\n', sum([1 2 3 4]));    % nonempty unchanged: 10
fprintf('meanv=%.0f\n', mean([2 4 6]));    % nonempty unchanged: 4
