function hash_out = fnv1a(byte_in, en, reset)
    %#codegen
    % hdl: port(byte_in, fi, unsigned, 8, 0)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % FNV-1a 32-bit hash (RFC 6234 algorithm). Streaming variant:
    % consumes one byte per cycle while `en` is asserted, updates
    % a 32-bit running hash. The classic FNV-1a step is:
    %   hash = (hash XOR byte) * FNV_PRIME
    %
    % Where FNV_PRIME = 16777619 = 0x01000193 and the initial
    % offset basis is 2166136261 = 0x811C9DC5.
    %
    % Tests:
    %   - 32-bit XOR-then-multiply chain on a persistent register
    %   - register snapshot pattern at i32 width
    %   - multiplication by a 32-bit constant (the synth tool
    %     should fold this into a shift-add tree post-CSD; the
    %     emitter just renders the literal `*`)

    FNV_PRIME = uint32(16777619);
    FNV_OFFSET = uint32(2166136261);

    persistent hash;
    if isempty(hash) || reset
        hash = FNV_OFFSET;
    end

    if en
        h = hash + uint32(0);                 % typed snapshot
        % Widen byte_in to u32 via constant accumulator (the
        % uint32(byte_in) runtime cast isn't synthesizable on a
        % runtime operand).
        b32 = uint32(0);
        if bitand(byte_in, uint8(1)) ~= 0;   b32 = b32 + uint32(1); end
        if bitand(byte_in, uint8(2)) ~= 0;   b32 = b32 + uint32(2); end
        if bitand(byte_in, uint8(4)) ~= 0;   b32 = b32 + uint32(4); end
        if bitand(byte_in, uint8(8)) ~= 0;   b32 = b32 + uint32(8); end
        if bitand(byte_in, uint8(16)) ~= 0;  b32 = b32 + uint32(16); end
        if bitand(byte_in, uint8(32)) ~= 0;  b32 = b32 + uint32(32); end
        if bitand(byte_in, uint8(64)) ~= 0;  b32 = b32 + uint32(64); end
        if bitand(byte_in, uint8(128)) ~= 0; b32 = b32 + uint32(128); end
        hash = bitxor(h, b32) * FNV_PRIME;
    end

    hash_out = hash;
end
