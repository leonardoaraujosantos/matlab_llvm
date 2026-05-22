% Tier B (any_shape): 3-D manipulation verbs on matlab_mat3.
% reshape / permute / ipermute / squeeze / cat(1|2,…) / dim-3 reductions /
% repmat-into-3rd.  Element order is the project's row-major/slice-major
% convention (documented), not MATLAB's column-major.

% B1: reshape vector -> 3-D, and 3-D -> 2-D (flat reinterpret)
A = reshape(1:24, 2, 3, 4);
fprintf('rs %.0fx%.0fx%.0f\n', size(A,1), size(A,2), size(A,3));
fprintf('rsv %.0f %.0f %.0f\n', A(1,1,1), A(1,1,2), A(2,3,4));
B2 = reshape(A, 6, 4);
fprintf('rs2d %.0fx%.0f n=%.0f\n', size(B2,1), size(B2,2), numel(B2));

% B5: reductions along dim 3 (channel reductions)
C = cat(3, ones(2,2)*1, ones(2,2)*2, ones(2,2)*4);
S = sum(C,3);     fprintf('sum3 %.0fx%.0f v=%.0f\n', size(S,1), size(S,2), S(1,1));
Mn = mean(C,3);   fprintf('mean3 v=%.1f\n', Mn(2,2));
Mx = max(C,[],3); fprintf('max3 v=%.0f\n', Mx(1,1));
Mi = min(C,[],3); fprintf('min3 v=%.0f\n', Mi(1,1));

% B2: permute + ipermute (round-trip)
P = permute(A, [3 1 2]);
fprintf('perm %.0fx%.0fx%.0f\n', size(P,1), size(P,2), size(P,3));
Q = ipermute(P, [3 1 2]);
fprintf('iperm %.0fx%.0fx%.0f eq=%.0f\n', size(Q,1), size(Q,2), size(Q,3), Q(2,3,4));

% B3: squeeze (drop a singleton leading dim)
R1 = ones(1,3,4);
Sq = squeeze(R1);
fprintf('sq %.0fx%.0f d=%.0f\n', size(Sq,1), size(Sq,2), ndims(Sq));

% B4: cat along dim 1 / dim 2 of 3-D blocks
D1 = cat(1, C, C);
fprintf('cat1 %.0fx%.0fx%.0f\n', size(D1,1), size(D1,2), size(D1,3));
D2 = cat(2, C, C);
fprintf('cat2 %.0fx%.0fx%.0f\n', size(D2,1), size(D2,2), size(D2,3));

% B6: repmat into the 3rd dim
T = repmat(ones(2,2)*5, 1, 1, 3);
fprintf('rep %.0fx%.0fx%.0f v=%.0f\n', size(T,1), size(T,2), size(T,3), T(1,1,3));
