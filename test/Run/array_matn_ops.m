% array_matn_ops.m — #93 item 2: a rank-N (matN) value reaching the
% general-purpose ops that previously only branched mat_is_3d. Verifies
% they handle a rank-N descriptor correctly instead of casting the matN
% header to matlab_mat (which read a garbage rows/cols / corrupted the heap).

A = zeros(2, 2, 2, 2, 2);     % rank-5, 32 elements
A(1, 1, 1, 1, 1) = 11;        % flat offset 0
A(2, 2, 2, 2, 2) = 55;        % flat offset 31

% reshape(matN, m, n) -> 2-D, row-major-extended flatten of all 32 elems.
R = reshape(A, 4, 8);
fprintf('matn_ops: reshape ndims=%.0f numel=%.0f\n', ndims(R), numel(R));

% A(i,j) fewer-subscript on a rank-5: i indexes dim 0, j collapses the
% trailing dims (dim 1 fastest).  A(1,1,1,1,1)=11 is at flat offset 0 ->
% A(1,1); A(2,2,2,2,2)=55 is at flat offset 31 -> A(2,16).
fprintf('matn_ops: A(1,1)  fewer-subscript = %.0f\n', A(1, 1));
fprintf('matn_ops: A(2,16) fewer-subscript = %.0f\n', A(2, 16));

fprintf('array_matn_ops: PASS\n');
