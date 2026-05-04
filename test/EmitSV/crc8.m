function crc = crc8(data_in, en, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % CRC-8 LFSR with polynomial 0x07 (x^8 + x^2 + x + 1).
    % One bit per cycle in; updates the persistent 8-bit CRC
    % register only when `en` is high. The standard XOR feedback
    % shape:
    %   - feedback = msb(crc) XOR data_in
    %   - shift crc left
    %   - xor in feedback at the polynomial taps (bits 0, 1, 2)
    %
    % Tests:
    %   - bitwise XOR feedback into a persistent register
    %   - high-bit extraction via bitand+shift
    %   - register self-update through a chain of bitwise ops

    persistent crc_reg;
    if isempty(crc_reg) || reset
        crc_reg = uint8(0);
    end

    if en
        % Snapshot the register into a local to break through the
        % runtime ABI's f64 return type (the bitwise lowering
        % requires both operands to have matching integer types,
        % which fails when one operand is the f64 from a
        % persistent_get_f64 call).
        cur = crc_reg + uint8(0);
        % Extract MSB of cur (bit 7).
        msb = bitand(bitshift(cur, -7), uint8(1));
        % data_in_u8: 0 or 1 via mutually-exclusive add (avoids
        % the conditional-store multi-source slot gap).
        data_in_u8 = uint8(0);
        if data_in
            data_in_u8 = uint8(1);
        end
        feedback = bitxor(msb, data_in_u8);
        % Shift left by 1 (clear LSB), then xor feedback into the
        % polynomial taps (bits 0, 1, 2). Polynomial 0x07 means
        % the feedback bit goes into bits 0, 1, and 2.
        shifted = bitand(bitshift(cur, 1), uint8(254));
        crc_reg = bitxor(shifted, feedback * uint8(7));
    end

    crc = crc_reg;
end
