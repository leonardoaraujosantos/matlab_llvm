function crc = crc32(data_in, en, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % hdl: port(crc, fi, unsigned, 32, 0)
    % cocotb: stimulus(reset, constant, 0)
    %
    % CRC-32 LFSR with the IEEE 802.3 polynomial 0x04C11DB7. One
    % bit per cycle in. The sequential bit-serial CRC update:
    %   feedback = msb(crc) XOR data_in
    %   crc = (crc << 1) XOR (feedback ? POLY : 0)
    %
    % Where POLY is just the polynomial taps masked to 32 bits.
    % Stress-tests the snapshot pattern at i32 width and a wider
    % XOR network than crc8.
    %
    % Tests:
    %   - 32-bit persistent register with bitwise updates
    %   - snapshot via `+ uint32(0)` at the wider width
    %   - branchless feedback dispatch (poly mask multiplied by
    %     the feedback bit)

    % IEEE 802.3 CRC-32 polynomial (excluding implicit x^32).
    POLY = uint32(79764919);   % 0x04C11DB7

    persistent crc_reg;
    if isempty(crc_reg) || reset
        crc_reg = uint32(0);   % all-zeros init (CRC residual depends
                                % on the user's start value;
                                % some standards prefer 0xFFFFFFFF)
    end

    if en
        cur = crc_reg + uint32(0);
        % Extract MSB of cur (bit 31).
        msb = bitand(bitshift(cur, -31), uint32(1));
        % Boolean → u32 conversion via branch.
        data_in_u32 = uint32(0);
        if data_in
            data_in_u32 = uint32(1);
        end
        feedback = bitxor(msb, data_in_u32);
        % crc << 1, then XOR poly when feedback was 1.
        shifted = bitshift(cur, 1);
        crc_reg = bitxor(shifted, feedback * POLY);
    end

    crc = crc_reg;
end
