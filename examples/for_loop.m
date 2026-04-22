% Sequential for-loops aren't lowered end-to-end yet (only parfor is),
% so we demonstrate loop-style work by expressing it with matrix ops
% that the compiler already knows how to run in parallel.
%
% sum(1..10) — the scalar accumulator version would be:
%     total = 0; for i = 1:10; total = total + i; end
% but the vectorised form produces the same answer without a loop:
disp('sum(1..10) =');
disp(sum(1:10));

% A 3x3 multiplication table built without a nested for-loop.
% (1:3)' is a column, (1:3) is a row; outer product = multiplication
% table.
disp('3x3 multiplication table:');
disp((1:3)' * (1:3));
