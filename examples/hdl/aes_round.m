function [out_word, ovfl] = aes_round(state_word, round_key)
    %#codegen
    % hdl: port(state_word, fi, unsigned, 32, 0)
    % hdl: port(round_key, fi, unsigned, 32, 0)
    % hdl: port(out_word, fi, unsigned, 32, 0)
    %
    % AES MixColumns + AddRoundKey for one 32-bit word (one column
    % of the 4x4 state). The full AES round also has SubBytes
    % (S-box) and ShiftRows; SubBytes needs a 256-entry ROM that
    % the current backend can't emit as a synthesized lookup, so
    % this design covers the bit-manipulation half of AES.
    %
    % MixColumns operates on each column [b0, b1, b2, b3] as:
    %   c0 = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3
    %   c1 = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3
    %   c2 = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3
    %   c3 = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3)
    %
    % Where xtime(x) = (x << 1) ^ (0x1B if msb(x) else 0). The
    % final result is XOR'd with the round key.
    %
    % Tests:
    %   - 32-bit byte-extract via bitand+bitshift
    %   - branchless conditional XOR (xtime via mul-by-bit)
    %   - dense XOR network (16 XORs in the column update)
    %   - bit-pack of 4 result bytes back into the 32-bit word
    %   - 32-bit XOR with round_key

    % Extract 4 bytes from state_word.
    b0 = bitand(state_word, uint32(255));
    b1 = bitand(bitshift(state_word, -8), uint32(255));
    b2 = bitand(bitshift(state_word, -16), uint32(255));
    b3 = bitand(bitshift(state_word, -24), uint32(255));

    % xtime helper inlined: x' = (x << 1) ^ (msb(x) ? 0x1B : 0).
    msb0 = bitand(bitshift(b0, -7), uint32(1));
    msb1 = bitand(bitshift(b1, -7), uint32(1));
    msb2 = bitand(bitshift(b2, -7), uint32(1));
    msb3 = bitand(bitshift(b3, -7), uint32(1));
    xt0 = bitxor(bitand(bitshift(b0, 1), uint32(255)), msb0 * uint32(27));
    xt1 = bitxor(bitand(bitshift(b1, 1), uint32(255)), msb1 * uint32(27));
    xt2 = bitxor(bitand(bitshift(b2, 1), uint32(255)), msb2 * uint32(27));
    xt3 = bitxor(bitand(bitshift(b3, 1), uint32(255)), msb3 * uint32(27));

    % MixColumns: each c_i is a fixed XOR of the bytes and xtimes.
    c0 = bitxor(bitxor(xt0, bitxor(xt1, b1)), bitxor(b2, b3));
    c1 = bitxor(bitxor(b0, bitxor(xt1, xt2)), bitxor(b2, b3));
    c2 = bitxor(bitxor(b0, b1), bitxor(bitxor(xt2, xt3), b3));
    c3 = bitxor(bitxor(xt0, b0), bitxor(bitxor(b1, b2), xt3));

    % Pack the four bytes back into the 32-bit word, then XOR
    % with the round key (AddRoundKey).
    packed = c0 + ...
             bitshift(c1, 8) + ...
             bitshift(c2, 16) + ...
             bitshift(c3, 24);
    out_word = bitxor(packed, round_key);

    % Status output: was the round-key fully consumed (i.e. did
    % the XOR change every byte). Just a witness output to keep
    % the design dual-output.
    ovfl = packed ~= out_word;
end
