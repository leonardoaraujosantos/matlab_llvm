% matmul3_batched — batched 2-D matmul over (M, K, B) * (K, N, B) -> (M, N, B).
%
% Verifies the batched product matches per-slice 2-D matmul.  Cumulative
% Frobenius squared-error is summed over both slices via cat3 + sum.

% Build A (2x3x2) and B (3x4x2) with distinct slices.
A = zeros(2, 3, 2);
B = zeros(3, 4, 2);
for i = 1:2
    for j = 1:3
        A(i, j, 1) = i + j;
        A(i, j, 2) = 2 * i - j;
    end
end
for i = 1:3
    for j = 1:4
        B(i, j, 1) = i * j;
        B(i, j, 2) = i - j;
    end
end

% Batched product (the new op).
C = matmul3(A, B);

% Reference: assemble the same per-slice 2-D matmuls into the same 3-D shape.
A1 = A(:, :, 1); A2 = A(:, :, 2);
B1 = B(:, :, 1); B2 = B(:, :, 2);
Ref1 = A1 * B1;
Ref2 = A2 * B2;
Ref  = cat(3, Ref1, Ref2);

% Element-wise difference + frobenius squared.
D   = C - Ref;
sse = sum(sum(sum(D .* D)));

fprintf('matmul3_batched: cumulative sse = %.6f\n', sse);

if sse < 1e-9
    fprintf('matmul3_batched: PASS\n');
else
    fprintf('matmul3_batched: FAIL\n');
end
