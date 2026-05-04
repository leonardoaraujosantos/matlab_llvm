function [y, valid] = cic_decimator(x, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 3rd-order CIC decimator with R=4 decimation rate.
    % Three integrators (running on the input rate) followed by
    % three combs (running on the decimated rate). Output gain
    % is R^N = 4^3 = 64; since we don't bit-trim, the bit-growth
    % bound is N*ceil(log2(R)) = 3*2 = 6 bits, so accumulators are
    % sized at 16+6 = 22 bits.
    %
    % This is a non-trivial test design (not in examples/hdl/'s
    % original 8) — exercises multiple persistent registers,
    % rate decimation via a downsample counter, and a final
    % saturating cast.

    % --- Integrators (3) — accumulate every sample ---
    persistent int1; persistent int2; persistent int3;
    % --- Comb delay registers — store the LAST decimated sample ---
    persistent comb1_d; persistent comb2_d; persistent comb3_d;
    % --- Downsample counter and stage outputs ---
    persistent ds_count;
    persistent comb1_out; persistent comb2_out;
    persistent y_reg; persistent valid_reg;

    if isempty(int1) || reset
        int1 = fi(0, 1, 22, 0);
        int2 = fi(0, 1, 22, 0);
        int3 = fi(0, 1, 22, 0);
        comb1_d = fi(0, 1, 22, 0);
        comb2_d = fi(0, 1, 22, 0);
        comb3_d = fi(0, 1, 22, 0);
        ds_count = uint8(0);
        comb1_out = fi(0, 1, 22, 0);
        comb2_out = fi(0, 1, 22, 0);
        y_reg = fi(0, 1, 16, 0);
        valid_reg = false;
    end

    % --- Integrator stage (every sample) ---
    int1 = int1 + fi(x, 1, 16, 0);
    int2 = int2 + int1;
    int3 = int3 + int2;

    % --- Downsample by 4 ---
    if ds_count == 3
        ds_count = uint8(0);
        % Comb stage on the decimated tap.
        c1 = int3 - comb1_d;
        comb1_d = int3;
        c2 = c1 - comb2_d;
        comb2_d = c1;
        c3 = c2 - comb3_d;
        comb3_d = c2;
        comb1_out = c1;
        comb2_out = c2;
        y_reg = fi(c3, 1, 16, 0, 'OverflowAction', 'Saturate');
        valid_reg = true;
    else
        ds_count = ds_count + uint8(1);
        valid_reg = false;
    end

    y = y_reg;
    valid = valid_reg;
end
