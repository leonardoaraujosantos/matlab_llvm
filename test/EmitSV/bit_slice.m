function [low_byte, high_byte, top_bit, low_word, three_bit, twelve_bit] = bit_slice(x)
    %#codegen
    % hdl: port(x, fi, unsigned, 32, 0)
    %
    % Bit-slice extension: `x(hi:lo)` extracts a constant bit-range
    % from a scalar integer, returning an unsigned value of the
    % rounded-up next-native width (1, 8, 16, 32, or 64). Native-width
    % slices render as a clean `x[hi:lo]` bit-select; non-aligned
    % widths render as `<resW>'(x[hi:lo])` with the size cast doing
    % the zero-extension.
    %
    % Tests:
    %   - aligned slices at every native width (1, 8, 16, 32)
    %   - high-byte / high-word extraction (lo > 0)
    %   - non-aligned widths (3-bit, 12-bit) — needs mask
    %   - port-arg source — bit-select renders as bare identifier

    low_byte   = x(7:0);     % uint8: bits 7..0 — `x[7:0]`
    high_byte  = x(31:24);   % uint8: high byte — `x[31:24]`
    top_bit    = x(31:31);   % logical: MSB — `x[31:31]`
    low_word   = x(15:0);    % uint16: low half — `x[15:0]`
    three_bit  = x(6:4);     % uint8 (3 meaningful bits) — needs mask
    twelve_bit = x(23:12);   % uint16 (12 meaningful bits) — needs mask
end
