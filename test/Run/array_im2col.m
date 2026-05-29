% array_im2col.m — Tier C im2col + GEMM-based conv2d_batch.
%
% Verifies:
%   (1) matlab_im2col_2d shape: (kH*kW*C) x (Hout*Wout*N)
%   (2) im2col reconstructs each input patch in the right column.
%   (3) The GEMM-rewritten matlab_conv2d_batch matches the explicit
%       7-deep loop's output bit-for-bit on a deterministic input.
%   (4) Trailing-singleton drop still routes correctly when K==1 or N==1.

% --- (1) im2col shape on a (3, 3, 2, 1) input, 2x2 kernel ---
X = zeros(3, 3, 2, 1);
for c = 1:2
    for h = 1:3
        for w = 1:3
            X(h, w, c, 1) = (c - 1) * 9 + (h - 1) * 3 + w;
        end
    end
end
Xc = im2col_2d(X, 2, 2);
% Expected shape: (2*2*2, 2*2*1) = (8, 4)
fprintf('array_im2col: Xc size = %.0f %.0f\n', size(Xc, 1), size(Xc, 2));

% --- (2) Spot-check a few cells ---
% Xc[c*kH*kW + kh*kW + kw, n*Hout*Wout + h*Wout + w] = X[h+kh, w+kw, c, n]
% Use 1-based MATLAB indexing — column for (n=0, h=0, w=0) is col 1.
% Channel 0, kh=0, kw=0  -> row 1  -> X(1,1,1,1) = 1
% Channel 1, kh=1, kw=1  -> row 8  -> X(2,2,2,1) = 9+4 = 14
fprintf('array_im2col: Xc(1,1)=%.0f  Xc(8,1)=%.0f\n', Xc(1, 1), Xc(8, 1));
% Column for (n=0, h=1, w=1) is col 4 (Hout=2, Wout=2, so 0*4 + 1*2 + 1 = 3 → col 4 in 1-based).
fprintf('array_im2col: Xc(1,4)=%.0f  Xc(8,4)=%.0f\n', Xc(1, 4), Xc(8, 4));

% --- (3) conv2d_batch — small deterministic case ---
% X: 4x4x2x2, W: 2x2x2x3.  Random-but-deterministic fill via index pattern.
X2 = zeros(4, 4, 2, 2);
for n = 1:2
    for c = 1:2
        for h = 1:4
            for w = 1:4
                X2(h, w, c, n) = (n - 1) * 32 + (c - 1) * 16 + (h - 1) * 4 + w;
            end
        end
    end
end
W2 = zeros(2, 2, 2, 3);
for k = 1:3
    for c = 1:2
        for kh = 1:2
            for kw = 1:2
                W2(kh, kw, c, k) = (k - 1) * 8 + (c - 1) * 4 + (kh - 1) * 2 + kw;
            end
        end
    end
end
Y2 = conv2d_batch(X2, W2);
fprintf('array_im2col: Y2 size = %.0f %.0f %.0f %.0f\n', ...
        size(Y2, 1), size(Y2, 2), size(Y2, 3), size(Y2, 4));
% Sample a few cells; values are deterministic with the indexing
% pattern above so any GEMM-vs-naive divergence would show.
fprintf('array_im2col: Y2(1,1,1,1) =%.0f\n', Y2(1, 1, 1, 1));
fprintf('array_im2col: Y2(3,3,2,1) =%.0f\n', Y2(3, 3, 2, 1));
fprintf('array_im2col: Y2(2,2,3,2) =%.0f\n', Y2(2, 2, 3, 2));

% --- (4) trailing-singleton drop, K==1, N==1 ---
X3 = zeros(3, 3, 1, 1);
X3(2, 2, 1, 1) = 5;
W3 = zeros(2, 2, 1, 1);
W3(1, 1, 1, 1) = 1; W3(2, 2, 1, 1) = 1;
Y3 = conv2d_batch(X3, W3);
% Output (2,2,1,1) -- trailing 1s drop to plain 2-D mat or mat3.
fprintf('array_im2col: Y3 ndims=%.0f size = %.0f %.0f\n', ...
        ndims(Y3), size(Y3, 1), size(Y3, 2));

fprintf('array_im2col: PASS\n');
