% array_matn_clone_on_assign.m — #102: `B = A` on a rank-N (matN) value must
% deep-copy the buffer, not shallow-alias the matlab_matN* pointer. The clone
% machinery (matlab_mat_clone_cow matN branch, #100) and the clone-on-assign
% gate (keyed on Shape::Rank::NDArray) were already in place; the gap was that
% Sema typed a rank-N constructor result as unknown-rank, so the gate — which
% excludes unknown rank — never fired and `B = A` aliased A's buffer.

A = zeros(2, 2, 2);          % rank-3 constructor (was unknown-rank pre-#102)
B = A;                       % must deep-copy, not alias
B(1, 1, 1) = 99;             % mutate B only
A(2, 2, 2) = 7;              % mutate A only

% If B aliased A, A(1,1,1) would read 99 and B(2,2,2) would read 7.
fprintf('clone: A(1,1,1) = %.0f\n', A(1, 1, 1));   % expect 0 (untouched)
fprintf('clone: B(1,1,1) = %.0f\n', B(1, 1, 1));   % expect 99
fprintf('clone: A(2,2,2) = %.0f\n', A(2, 2, 2));   % expect 7
fprintf('clone: B(2,2,2) = %.0f\n', B(2, 2, 2));   % expect 0 (untouched)

fprintf('array_matn_clone_on_assign: PASS\n');
