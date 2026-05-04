function [pos, valid] = leading_zero_detector(x)
    %#codegen
    % hdl: port(x, fi, unsigned, 16, 0)
    %
    % Leading-zero detector for a 16-bit input. `pos` reports the
    % bit index of the highest set bit (15 if MSB set, 0 if only
    % LSB set); `valid` is high when at least one bit is set, low
    % when x is zero.
    %
    % The reverse-priority chain to priority_encoder.m: priority
    % encoder finds the LOWEST set bit; LZD finds the HIGHEST.
    % Both lower to a priority mux at synthesis. Tests:
    %   - reverse-direction if/elseif chain (MSB-first)
    %   - 16-branch chain length

    if bitand(x, uint16(32768)) ~= 0
        pos = uint8(15); valid = true;
    elseif bitand(x, uint16(16384)) ~= 0
        pos = uint8(14); valid = true;
    elseif bitand(x, uint16(8192)) ~= 0
        pos = uint8(13); valid = true;
    elseif bitand(x, uint16(4096)) ~= 0
        pos = uint8(12); valid = true;
    elseif bitand(x, uint16(2048)) ~= 0
        pos = uint8(11); valid = true;
    elseif bitand(x, uint16(1024)) ~= 0
        pos = uint8(10); valid = true;
    elseif bitand(x, uint16(512)) ~= 0
        pos = uint8(9); valid = true;
    elseif bitand(x, uint16(256)) ~= 0
        pos = uint8(8); valid = true;
    elseif bitand(x, uint16(128)) ~= 0
        pos = uint8(7); valid = true;
    elseif bitand(x, uint16(64)) ~= 0
        pos = uint8(6); valid = true;
    elseif bitand(x, uint16(32)) ~= 0
        pos = uint8(5); valid = true;
    elseif bitand(x, uint16(16)) ~= 0
        pos = uint8(4); valid = true;
    elseif bitand(x, uint16(8)) ~= 0
        pos = uint8(3); valid = true;
    elseif bitand(x, uint16(4)) ~= 0
        pos = uint8(2); valid = true;
    elseif bitand(x, uint16(2)) ~= 0
        pos = uint8(1); valid = true;
    elseif bitand(x, uint16(1)) ~= 0
        pos = uint8(0); valid = true;
    else
        pos = uint8(0); valid = false;
    end
end
