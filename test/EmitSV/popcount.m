function n = popcount(x)
    %#codegen
    % hdl: port(x, fi, unsigned, 16, 0)
    %
    % Population count (Hamming weight) of a 16-bit input. Sums
    % each individual bit. The MATLAB source uses bitand + shift
    % at constant positions; the SV emitter renders as a chain of
    % conditional adds that synthesizes to a balanced adder tree.
    %
    % Tests:
    %   - bit-extraction via `bitand(bitshift(x, -K), 1)` with
    %     constant K (post-bitshift-lowering fix)
    %   - 16-element accumulator pattern, all i8-typed, into a
    %     5-bit-wide output (max value 16)
    %   - integer-typed initial accumulator value followed by
    %     conditional adds that all flow into the output

    n = uint8(0);
    if bitand(x, uint16(1))     ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -1),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -2),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -3),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -4),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -5),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -6),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -7),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -8),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -9),  uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -10), uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -11), uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -12), uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -13), uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -14), uint16(1)) ~= 0; n = n + uint8(1); end
    if bitand(bitshift(x, -15), uint16(1)) ~= 0; n = n + uint8(1); end
end
