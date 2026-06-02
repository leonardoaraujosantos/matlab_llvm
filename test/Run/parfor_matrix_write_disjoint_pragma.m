% Issue #33 Phase 3b — `% matlab_llvm: write-disjoint(A, j)` escape hatch.
% A parfor that element-writes a captured matrix through a permutation
% index (`A(perm(j)) = ...`).  The structural disjointness check can't
% prove the index injective in the loop variable, so it would reject the
% write.  The pragma asserts the writes don't alias across iterations —
% true here because `perm` is a permutation, so each iteration targets a
% distinct cell — and the outliner trusts it.
%
% Every printed value is an order-independent aggregate, so the result is
% deterministic regardless of thread interleaving and bit-exact vs. a
% sequential for-loop.

% Permutation-indexed scalar write.
N = 5;
A = zeros(1, N);
perm = [3 1 5 2 4];
% matlab_llvm: write-disjoint(A, j)
parfor j = 1:N
    A(perm(j)) = j * 10;
end
% A(perm(j)) = j*10  ->  A = [20 40 10 50 30]; sum = 150.
fprintf('perm_sum = %d\n', sum(A));

% Indirect row target via a permutation, leading index derived from the
% loop variable but not the bare IV.
P = zeros(4, 4);
order = [4 3 2 1];
% matlab_llvm: write-disjoint(P, i)
parfor i = 1:4
    P(order(i), :) = i;
end
% Row order(i) holds four copies of i: 4*(1+2+3+4) = 40.
fprintf('perm_row_sum = %d\n', sum(P(:)));
