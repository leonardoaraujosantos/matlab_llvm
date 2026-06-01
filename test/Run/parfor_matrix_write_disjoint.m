% Issue #33 Phase 3b — write-disjoint matrix captures.  A matrix
% allocated outside the parfor body and element-written inside, one
% distinct cell per iteration.  The capture machinery (#51 Phase 3a)
% passes the matrix descriptor by shared pointer through state[]; the
% runtime slice-store helpers mutate it in place, so the writes are
% visible after the loop without any write-back.
%
% Soundness gate (OutlineParfor): the write is accepted only because
% each store is indexed by the loop variable, so distinct iterations
% touch distinct cells and can't race.  Non-disjoint writes (constant
% or non-injective index) are rejected at compile time instead.
%
% Every printed value is an order-independent aggregate, so the result
% is deterministic regardless of thread interleaving.

% 1-D scalar write, one element per iteration.
N = 10;
row_totals = zeros(N, 1);
parfor j = 1:N
    row_totals(j) = j * 2;
end
% 2 + 4 + ... + 20 = 110.
fprintf('row_totals_sum = %d\n', sum(row_totals));

% 2-D diagonal write — both indices are the loop variable.
M = zeros(4, 4);
parfor i = 1:4
    M(i, i) = i * 10;
end
% 10 + 20 + 30 + 40 = 100.
fprintf('diag_sum = %d\n', sum(M(:)));

% 2-D whole-row write — leading index is the loop variable, columns ':'.
R = zeros(4, 4);
parfor i = 1:4
    R(i, :) = i;
end
% Each row i holds four copies of i: 4*(1+2+3+4) = 40.
fprintf('row_sum = %d\n', sum(R(:)));

% Read one captured matrix, write a disjoint slot of another.
A = magic(5);
out = zeros(5, 1);
parfor i = 1:5
    out(i) = A(i, i) * 2;
end
% diag(magic(5)) = [17 5 13 21 9]; sum*2 = 130.
fprintf('out_sum = %d\n', sum(out));
