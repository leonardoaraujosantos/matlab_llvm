function code = hamming74(data)
    %#codegen
    % hdl: port(data, fi, unsigned, 8, 0)
    %
    % Hamming(7, 4) encoder. Takes 4 data bits in the low nibble of
    % the 8-bit input (high nibble ignored) and produces a 7-bit
    % codeword: parity-bit / parity-bit / data-bit / parity-bit /
    % data-bit / data-bit / data-bit. Output codeword in the low 7
    % bits of an 8-bit field; bit 7 is zero.
    %
    % Tests:
    %   - XOR networks across bit positions (multiple bitand+xor on
    %     the same input)
    %   - bit-packing back into a multi-bit output via OR-of-shifts
    %   - 8-bit input/output ports

    % Extract 4 data bits.
    d1 = bitand(data, uint8(1));                       % bit 0
    d2 = bitand(bitshift(data, -1), uint8(1));         % bit 1
    d3 = bitand(bitshift(data, -2), uint8(1));         % bit 2
    d4 = bitand(bitshift(data, -3), uint8(1));         % bit 3

    % Hamming(7,4) parity bits per the standard tables.
    p1 = bitxor(bitxor(d1, d2), d4);
    p2 = bitxor(bitxor(d1, d3), d4);
    p3 = bitxor(bitxor(d2, d3), d4);

    % Pack the codeword: p1 p2 d1 p3 d2 d3 d4 (LSB-first).
    code = p1 + ...
           bitshift(p2, 1) + ...
           bitshift(d1, 2) + ...
           bitshift(p3, 3) + ...
           bitshift(d2, 4) + ...
           bitshift(d3, 5) + ...
           bitshift(d4, 6);
end
