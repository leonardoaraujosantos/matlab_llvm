% #191 P2.2 — a matrix reduction's result carries a concrete 1xN row shape
% into a downstream user-function call argument (was Any, which poisoned
% inter-procedural arg precision). Also a regression guard for the scalar /
% multi-output / elementwise reduction forms that must stay on the runtime
% (boxed-ptr) path and must NOT be mistyped as unboxed scalars.
M = [1 2 3; 4 5 6];
r = rowmean(mean(M));        % mean(M) -> 1x3 row, fed into a user fn arg
fprintf('r %.4f\n', r);

[v, k] = max([3 9 2]);       % multi-output max: value + index
fprintf('mx %.0f %.0f\n', v, k);

e = max([1 -2 3], 0);        % elementwise max(x, 0)
fprintf('e %.0f %.0f %.0f\n', e(1), e(2), e(3));

s = sum([4 5 6]);            % scalar reduction (runtime path)
fprintf('s %.0f\n', s);

cs = sum(M);                 % matrix reduction -> 1x3 row
fprintf('cs %.0f %.0f %.0f\n', cs(1), cs(2), cs(3));

function m = rowmean(row)
  m = mean(row);             % mean over the 1x3 row
end
