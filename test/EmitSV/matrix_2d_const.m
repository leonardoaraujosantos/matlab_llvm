function [a, b, c, d] = matrix_2d_const(in_a, in_b, in_c, in_d, reset)
    %#codegen
    % hdl: port(in_a, fi, signed, 16, 0)
    % hdl: port(in_b, fi, signed, 16, 0)
    % hdl: port(in_c, fi, signed, 16, 0)
    % hdl: port(in_d, fi, signed, 16, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 2x2 persistent fi-matrix with constant (row, col) reads /
    % writes. Tier-2 A: Stage F now flattens 2-D `fi(zeros(M, N))`
    % to row-major linear storage with `(i-1)*N + (j-1)` index
    % math, so `mat(1, 2) = v` lowers to a write at flat index 1
    % and `mat(2, 1)` reads flat index 2. The RAM-inference v1
    % pass collapses the 4 split scalars into a single
    % `logic [W-1:0] mat [4]` array.
    persistent mat;
    if isempty(mat) || reset
        mat = fi(zeros(2, 2), 1, 16, 0);
    end
    mat(1, 1) = in_a;
    mat(1, 2) = in_b;
    mat(2, 1) = in_c;
    mat(2, 2) = in_d;
    a = mat(1, 1);
    b = mat(1, 2);
    c = mat(2, 1);
    d = mat(2, 2);
end
