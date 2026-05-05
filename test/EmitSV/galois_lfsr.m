function rand_out = galois_lfsr(en, reset)
    %#codegen
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 16-bit Galois LFSR with the standard maximum-length polynomial
    % 0xB400 (x^16 + x^14 + x^13 + x^11 + 1). Outputs a new pseudo-
    % random 16-bit value every cycle while `en` is asserted; cycles
    % through all 65535 non-zero states before repeating.
    %
    % Tests:
    %   - wide (16-bit) XOR feedback into a persistent register
    %   - bitwise XOR of register snapshot with a polynomial mask
    %   - the snapshot pattern at i16 width (regression-testing the
    %     uint16(0)-typing fix from the prior round)

    persistent state;
    if isempty(state) || reset
        state = uint16(1);   % seed; all-zero state would be a fixed point
    end

    if en
        % Snapshot to break through the runtime f64 ABI.
        s = state + uint16(0);
        % Galois LFSR: shift right one bit; if the LSB was 1, XOR
        % the polynomial mask into the shifted state.
        if bitand(s, uint16(1)) ~= 0
            state = bitxor(bitshift(s, -1), uint16(46080));
        else
            state = bitshift(s, -1);
        end
    end

    rand_out = state;
end
