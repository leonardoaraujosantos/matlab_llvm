% `~` ignore-output in a multi-return LHS, plus the value+index second
% output of min / max / sort.
[~, k] = min([5 2 8]);
fprintf('mink %.0f\n', k);

[v, i] = max([3 9 2]);
fprintf('max %.0f i %.0f\n', v, i);

[s, p] = sort([3 1 2]);
fprintf('sort %.0f %.0f %.0f\n', s(1), s(2), s(3));
fprintf('sortidx %.0f %.0f %.0f\n', p(1), p(2), p(3));

[~, c] = size([1 2 3; 4 5 6]);
fprintf('cols %.0f\n', c);

% the index output used to subscript another array
w = [10 20 30];
[~, j] = min([5 2 8]);
fprintf('wj %.0f\n', w(j));
