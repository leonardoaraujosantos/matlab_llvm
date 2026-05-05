function [x_out, y_out, z_out] = cordic_pipe(x_in, y_in, z_in, reset)
    %#codegen
    % hdl: port(x_in, fi, signed, 16, 0)
    % hdl: port(y_in, fi, signed, 16, 0)
    % hdl: port(z_in, fi, signed, 16, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 4-stage pipelined CORDIC. Each stage performs one rotation
    % iteration with a precomputed atan(2^-k) constant. Stage k
    % shifts the (x, y) pair by k bits and adds/subtracts based
    % on the sign of z. Inter-stage registers latch the (x, y, z)
    % triple between iterations.
    %
    % Atan constants (Q15 fixed-point representation):
    %   atan(2^0)  = 0.7854  → 25735
    %   atan(2^-1) = 0.4636  → 15192
    %   atan(2^-2) = 0.2450  →  8027
    %   atan(2^-3) = 0.1244  →  4076
    %
    % Tests:
    %   - 12 chained persistent registers (4 stages × 3 datapaths)
    %   - per-stage signed conditional add/subtract under sign-of-z
    %   - per-stage constant arithmetic-right shift
    %   - canonical pipelined-DSP pattern (compute and forward to
    %     the next stage's register on each cycle)

    % --- Stage 1 registers (latched from input on edge) ---
    persistent x1; persistent y1; persistent z1;
    % --- Stage 2 registers ---
    persistent x2; persistent y2; persistent z2;
    % --- Stage 3 registers ---
    persistent x3; persistent y3; persistent z3;
    % --- Stage 4 (final) registers ---
    persistent x4; persistent y4; persistent z4;

    if isempty(x1) || reset; x1 = int16(0); end
    if isempty(y1) || reset; y1 = int16(0); end
    if isempty(z1) || reset; z1 = int16(0); end
    if isempty(x2) || reset; x2 = int16(0); end
    if isempty(y2) || reset; y2 = int16(0); end
    if isempty(z2) || reset; z2 = int16(0); end
    if isempty(x3) || reset; x3 = int16(0); end
    if isempty(y3) || reset; y3 = int16(0); end
    if isempty(z3) || reset; z3 = int16(0); end
    if isempty(x4) || reset; x4 = int16(0); end
    if isempty(y4) || reset; y4 = int16(0); end
    if isempty(z4) || reset; z4 = int16(0); end

    % Snapshot every persist-get into a typed local first. The
    % runtime ABI returns f64 from each get; without snapshots,
    % the bitshift in stages 2/3 sees a raw f64 and the lowering
    % bails. Stage 1 doesn't shift, but we snapshot for symmetry.
    s1x = x1 + int16(0); s1y = y1 + int16(0); s1z = z1 + int16(0);
    s2x = x2 + int16(0); s2y = y2 + int16(0); s2z = z2 + int16(0);
    s3x = x3 + int16(0); s3y = y3 + int16(0); s3z = z3 + int16(0);

    % --- Stage 1: shift by 0, atan = 25735 ---
    if s1z >= 0
        x2 = s1x - s1y;
        y2 = s1y + s1x;
        z2 = s1z - int16(25735);
    else
        x2 = s1x + s1y;
        y2 = s1y - s1x;
        z2 = s1z + int16(25735);
    end

    % --- Stage 2: shift by 1, atan = 15192 ---
    if s2z >= 0
        x3 = s2x - bitshift(s2y, -1);
        y3 = s2y + bitshift(s2x, -1);
        z3 = s2z - int16(15192);
    else
        x3 = s2x + bitshift(s2y, -1);
        y3 = s2y - bitshift(s2x, -1);
        z3 = s2z + int16(15192);
    end

    % --- Stage 3: shift by 2, atan = 8027 ---
    if s3z >= 0
        x4 = s3x - bitshift(s3y, -2);
        y4 = s3y + bitshift(s3x, -2);
        z4 = s3z - int16(8027);
    else
        x4 = s3x + bitshift(s3y, -2);
        y4 = s3y - bitshift(s3x, -2);
        z4 = s3z + int16(8027);
    end

    % --- Pipeline input latching ---
    x1 = x_in;
    y1 = y_in;
    z1 = z_in;

    % --- Output (final stage register) ---
    x_out = x4;
    y_out = y4;
    z_out = z4;
end
