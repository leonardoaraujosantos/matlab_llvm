% Convolution demo — exercises conv (1-D) and conv2 (2-D), full shape.
%
% conv(u, v) returns a vector of length numel(u) + numel(v) - 1.
% conv2(A, B) returns an (m1+m2-1) x (n1+n2-1) matrix.

% --- 1-D: row-vector convolution -----------------------------------------
u = [1 2 3 4 5];
disp('u =');
disp(u);

% Smoothing kernel — a 3-tap moving sum. Length 5 + 3 - 1 = 7.
disp('conv(u, [1 1 1]) — 3-tap moving sum:');
disp(conv(u, [1 1 1]));

% Polynomial multiplication: (1 + 2x + 3x^2) * (1 + x) = 1 + 3x + 5x^2 + 3x^3
disp('conv([1 2 3], [1 1]) — polynomial product:');
disp(conv([1 2 3], [1 1]));

% Edge-detector kernel [1 -1] — first differences, plus boundary terms.
disp('conv(u, [1 -1]) — first differences (full):');
disp(conv(u, [1 -1]));

% --- 2-D: image convolution ----------------------------------------------
A = [1 2 3;
     4 5 6;
     7 8 9];
disp('A =');
disp(A);

% 2x2 box kernel — output is 4x4 (3+2-1) and each entry sums a 2x2 window
% with implicit zero padding around A.
disp('conv2(A, ones(2,2)) — 2x2 box, full shape (4x4):');
disp(conv2(A, ones(2, 2)));

% Outer-product check: conv2 of two 1-D kernels equals the outer product
% of their 1-D convolutions. Here [1 1] (*) [1 1]' should give ones(2,2).
disp('conv2([1 1], [1; 1]) — outer product (should be ones(2,2)):');
disp(conv2([1 1], [1; 1]));

% 3x3 Sobel-x kernel applied to A.
sobelX = [-1 0 1;
          -2 0 2;
          -1 0 1];
disp('conv2(A, sobelX) — 5x5 result:');
disp(conv2(A, sobelX));
