% array_tierc_rank5.m — Tier C #93: rank>=5 explicit element store + read.
% The runtime subscriptN helpers (matlab_subscriptN_s / _pstore_s) are
% generic to 16 dims; this verifies the variadic Lowering arm wires
% A(i,j,k,l,m[,...]) = v and the matching scalar read back.

% Rank-5: 2x2x2x2x2 (32 elements).  Store at a few cells, read back.
A = zeros(2, 2, 2, 2, 2);
A(1, 1, 1, 1, 1) = 11;
A(2, 2, 2, 2, 2) = 55;
A(1, 2, 1, 2, 1) = 33;
fprintf('rank5: A(1,1,1,1,1)=%.0f A(2,2,2,2,2)=%.0f A(1,2,1,2,1)=%.0f\n', ...
        A(1, 1, 1, 1, 1), A(2, 2, 2, 2, 2), A(1, 2, 1, 2, 1));

% An untouched cell still reads zero.
fprintf('rank5: A(2,1,2,1,2)=%.0f\n', A(2, 1, 2, 1, 2));

% ones(...) rank-5 reads back 1 at every cell.
B = ones(2, 2, 2, 2, 2);
fprintf('rank5: ones B(1,1,1,1,1)=%.0f B(2,2,2,2,2)=%.0f\n', ...
        B(1, 1, 1, 1, 1), B(2, 2, 2, 2, 2));

% Rank-6: 2x2x2x2x2x2 (64 elements).  Store + read at the corners.
C = zeros(2, 2, 2, 2, 2, 2);
C(1, 1, 1, 1, 1, 1) = 7;
C(2, 2, 2, 2, 2, 2) = 64;
fprintf('rank6: C(1,1,1,1,1,1)=%.0f C(2,2,2,2,2,2)=%.0f\n', ...
        C(1, 1, 1, 1, 1, 1), C(2, 2, 2, 2, 2, 2));
fprintf('rank6: ndims(C)=%.0f numel(C)=%.0f\n', ndims(C), numel(C));

fprintf('array_tierc_rank5: PASS\n');
