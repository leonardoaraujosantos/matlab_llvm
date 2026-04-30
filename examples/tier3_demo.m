% Tier-3 builtins demo: rank, cond, null, orth, imfilter, padarray,
% interp2, upsample, downsample.
%
% This file also exercises the new scalar-promotion fallback —
% conv(u, 2), polyval(p, 5) etc. now auto-box scalars instead of
% requiring [2] / [5] workarounds.

% --- rank / cond ---------------------------------------------------------
A = [1 2 3;
     4 5 6;
     7 8 9];
disp('A = magic-ish singular matrix:');
disp(A);

% rank(A) is 2 (rows are not linearly independent).
disp('rank(A) — expect 2:');
disp(rank(A));

% Full-rank 3x3.
B = [1 0 0;
     0 2 0;
     0 0 3];
disp('rank(diag([1 2 3])) — expect 3:');
disp(rank(B));
disp('cond(diag([1 2 3])) — expect 3 (3/1):');
disp(cond(B));

% --- null / orth ---------------------------------------------------------
% Singular A above has rank 2, so its null space is 1-dimensional.
disp('null(A) — single column basis vector:');
disp(null(A));
% A * null(A) should be ~ 0.
disp('A * null(A) — should be ~ 0:');
disp(A * null(A));

% orth(A) — 2 columns spanning col(A).
disp('orth(A) — two orthonormal columns:');
disp(orth(A));

% --- imfilter / padarray -------------------------------------------------
img = [1 2 3;
       4 5 6;
       7 8 9];
% 3x3 averaging filter (box) — output size matches input ('same' shape).
h = ones(3, 3) / 9;
disp('imfilter(img, ones(3,3)/9) — same-size box average:');
disp(imfilter(img, h));

disp('padarray(img, [1 1]) — zero-pad by 1 row + 1 col on each side:');
disp(padarray(img, [1 1]));

% --- interp2 -------------------------------------------------------------
% Sample z = x + 10*y on a 3x3 grid, then query (1.5, 0.5):
% expected = 1.5 + 10*0.5 = 6.5.
xv = [0 1 2];
yv = [0; 1; 2];
V  = [0 1 2;
      10 11 12;
      20 21 22];
disp('interp2 sample at (1.5, 0.5) — expect 6.5:');
disp(interp2(xv, yv, V, [1.5], [0.5]));

% --- upsample / downsample ----------------------------------------------
disp('upsample([1 2 3 4], 3) — insert 2 zeros between samples:');
disp(upsample([1 2 3 4], 3));

disp('downsample([10 20 30 40 50 60], 2) — every 2nd sample:');
disp(downsample([10 20 30 40 50 60], 2));

% --- Scalar-promotion fallback ------------------------------------------
% Previously these collapsed to f64 and missed dispatch. Now they auto-box.
disp('conv([1 2 3], 2) — scalar gain via auto-box; expect [2 4 6]:');
disp(conv([1 2 3], 2));

disp('filter([1 0], 1, [1 2 3 4]) — identity filter via boxed scalar:');
disp(filter([1 0], 1, [1 2 3 4]));

disp('polyval([1 -3 2], 5) — scalar evaluation via auto-box; expect 12:');
disp(polyval([1 -3 2], 5));
