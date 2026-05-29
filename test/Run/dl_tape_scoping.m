% dl_tape_scoping.m — verify dlreset() truncates the dlnet tape across
% training iters so memory + per-iter wall time stay bounded.

% Initial tape size before any dlarray work.
sz0 = dltape_size(0);
fprintf('dl_tape_scoping: initial tape = %.0f\n', sz0);

% Iter 1: a minimal forward + backward that records 3 nodes
% (1 leaf for X, 1 OP_TIMES, 1 OP_SUM).
X1 = dlarray(ones(3, 3));
Y1 = X1 .* X1;
L1 = sum(Y1);
g1 = dlgradient(L1, X1);
sz1 = dltape_size(0);
fprintf('dl_tape_scoping: after iter1 tape = %.0f\n', sz1);

% Iter 2 WITHOUT reset — tape piles on.
X2 = dlarray(ones(3, 3));
Y2 = X2 .* X2;
L2 = sum(Y2);
g2 = dlgradient(L2, X2);
sz2 = dltape_size(0);
fprintf('dl_tape_scoping: after iter2 (no reset) tape = %.0f (grew)\n', sz2);

% Reset between iters.
dlreset();
sz3 = dltape_size(0);
fprintf('dl_tape_scoping: after dlreset() tape = %.0f\n', sz3);

% Iter 3 WITH reset.
X3 = dlarray(ones(3, 3));
Y3 = X3 .* X3;
L3 = sum(Y3);
g3 = dlgradient(L3, X3);
sz4 = dltape_size(0);
fprintf('dl_tape_scoping: after iter3 (with reset) tape = %.0f\n', sz4);

% Gradient invariance — g1 and g3 should be identical (= 2*X = 2*ones).
diff = 0;
for i = 1:3
    for j = 1:3
        diff = diff + abs(g1(i, j) - g3(i, j));
    end
end
fprintf('dl_tape_scoping: |g1 - g3| sum = %.4f\n', diff);

if sz4 <= sz1 + 1
    fprintf('dl_tape_scoping: PASS (reset bounded the tape)\n');
else
    fprintf('dl_tape_scoping: FAIL\n');
end
