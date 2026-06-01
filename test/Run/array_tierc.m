% array_tierc.m — Tier C gating: rank-N tensor descriptor (matlab_matN).
% Verifies the new descriptor + constructors + size/numel/ndims polymorphism
% on rank-4 / rank-5 inputs, plus that trailing-singleton drop still routes
% legacy 3-D and 2-D paths through their fast descriptors.

% Rank-4: 2x3x4x5 — should be a true matN.
A = zeros(2, 3, 4, 5);
fprintf('array_tierc: ndims(A)=%.0f numel(A)=%.0f\n', ndims(A), numel(A));
fprintf('array_tierc: size(A,1..4) = %.0f %.0f %.0f %.0f\n', ...
        size(A, 1), size(A, 2), size(A, 3), size(A, 4));
fprintf('array_tierc: size(A,5)    = %.0f\n', size(A, 5));   % trailing => 1

% Rank-5: 2x2x2x2x2 ones, 32 elements all 1.
B = ones(2, 2, 2, 2, 2);
fprintf('array_tierc: ndims(B)=%.0f numel(B)=%.0f\n', ndims(B), numel(B));

% Trailing-singleton drop:
%   zeros(2,3,4,1) -> matN with effective ndims=3 -> matlab_mat3
%   zeros(2,3,1,1) -> matlab_mat (2-D)
C = zeros(2, 3, 4, 1);
fprintf('array_tierc: zeros(2,3,4,1) ndims=%.0f size3=%.0f\n', ...
        ndims(C), size(C, 3));
D = zeros(2, 3, 1, 1);
fprintf('array_tierc: zeros(2,3,1,1) ndims=%.0f size2=%.0f\n', ...
        ndims(D), size(D, 2));

% Rank-4 element store + read: A(i,j,k,l) = v ; A(i,j,k,l) round-trip.
A2 = zeros(2, 3, 4, 5);
A2(1, 1, 1, 1) = 11;
A2(2, 3, 4, 5) = 99;
A2(1, 2, 3, 4) = 23;
fprintf('array_tierc: A2(1,1,1,1)=%.0f A2(2,3,4,5)=%.0f A2(1,2,3,4)=%.0f\n', ...
        A2(1, 1, 1, 1), A2(2, 3, 4, 5), A2(1, 2, 3, 4));

% Rank-5 construction works here; per-element store/read for arities >= 5
% is covered by the variadic Lowering arm in array_tierc_rank5.m (#93).

% reshape(A, d1,d2,d3,d4) -> rank-4 result.  Source can be any rank;
% here we start from a 2x12 (24 elements) and reshape to 2x3x2x2.
M = zeros(2, 12);
M(1, 1) = 100;
M(2, 12) = 200;
R = reshape(M, 2, 3, 2, 2);
fprintf('array_tierc: reshape ndims=%.0f numel=%.0f\n', ndims(R), numel(R));
fprintf('array_tierc: reshape size = %.0f %.0f %.0f %.0f\n', ...
        size(R, 1), size(R, 2), size(R, 3), size(R, 4));

% squeeze: drop singleton dim from a 4-D 2x1x3x1 -> 2x3.
S0 = zeros(2, 1, 3, 1);
S0(1, 1, 2, 1) = 5;
S0(2, 1, 3, 1) = 7;
S = squeeze(S0);
fprintf('array_tierc: squeeze ndims=%.0f size = %.0f %.0f\n', ...
        ndims(S), size(S, 1), size(S, 2));

% permute: rank-4 axis swap [2 1 4 3] on a 2x3x4x5 array.
P0 = zeros(2, 3, 4, 5);
P0(1, 1, 1, 1) = 42;
P0(2, 3, 4, 5) = 88;
P = permute(P0, [2 1 4 3]);   % new shape 3x2x5x4
fprintf('array_tierc: permute ndims=%.0f size = %.0f %.0f %.0f %.0f\n', ...
        ndims(P), size(P, 1), size(P, 2), size(P, 3), size(P, 4));
fprintf('array_tierc: permute P(1,1,1,1)=%.0f P(3,2,5,4)=%.0f\n', ...
        P(1, 1, 1, 1), P(3, 2, 5, 4));

% Elementwise on matN: matrix-scalar + matrix-matrix.
E1 = ones(2, 2, 2, 2);          % all-ones rank-4
E2 = 3 * E1;                    % matrix-scalar (s * matN)
fprintf('array_tierc: 3*ones(2,2,2,2): E2(1,1,1,1)=%.0f E2(2,2,2,2)=%.0f\n', ...
        E2(1, 1, 1, 1), E2(2, 2, 2, 2));
E3 = E2 + E1;                   % matN + matN, same shape
fprintf('array_tierc: matN+matN E3(1,1,1,1)=%.0f E3(2,2,2,2)=%.0f\n', ...
        E3(1, 1, 1, 1), E3(2, 2, 2, 2));

% sum reduction on a rank-4 array along each axis.  Source is a 2x2x2x2
% of 1s; reducing along ANY axis gives a rank-4 result with that dim=1,
% which collapses if trailing.  Total numel(reduced) = numel(original)/2.
S4 = ones(2, 2, 2, 2);
R1 = sum(S4, 1);                % 1x2x2x2 -> collapses trailing? dim1=1 not trailing
R2 = sum(S4, 4);                % 2x2x2x1 -> trailing-singleton -> rank 3
fprintf('array_tierc: sum(S4,1) ndims=%.0f numel=%.0f total=%.0f\n', ...
        ndims(R1), numel(R1), sum(sum(sum(R1, 4), 3), 2));
fprintf('array_tierc: sum(S4,4) ndims=%.0f numel=%.0f\n', ...
        ndims(R2), numel(R2));

% disp(matN) — page-by-page render with the new C5 path.  Build a tiny
% (2,2,2,2) tensor with a single non-zero cell so the output is bounded.
D0 = zeros(2, 2, 2, 2);
D0(2, 2, 2, 2) = 42;
disp(D0);

fprintf('array_tierc: PASS\n');
