function [valid, idx] = priority_encoder(req)
    %#codegen
    % hdl: port(req, fi, unsigned, 8, 0)
    %
    % 8-input priority encoder. `req` carries 8 request lines,
    % LSB-first priority. Output `idx` is the bit position of the
    % lowest set bit (0..7); `valid` is high when any bit is set.
    %
    % The canonical priority-encoder pattern: long if/elseif/else
    % chain testing each bit from lowest priority. Real designs
    % often use casez with don't-care bits, but MATLAB doesn't
    % have don't-care literals; the chain form synthesizes to the
    % same priority mux.
    %
    % Tests:
    %   - long if/elseif/else chain (10 branches)
    %   - non-mutually-exclusive conditions (priority semantics
    %     come from the chain order, not from the conditions)
    %   - dual-output design where both outputs are always written
    %     under each branch

    if bitand(req, uint8(1)) ~= 0
        idx = uint8(0); valid = true;
    elseif bitand(req, uint8(2)) ~= 0
        idx = uint8(1); valid = true;
    elseif bitand(req, uint8(4)) ~= 0
        idx = uint8(2); valid = true;
    elseif bitand(req, uint8(8)) ~= 0
        idx = uint8(3); valid = true;
    elseif bitand(req, uint8(16)) ~= 0
        idx = uint8(4); valid = true;
    elseif bitand(req, uint8(32)) ~= 0
        idx = uint8(5); valid = true;
    elseif bitand(req, uint8(64)) ~= 0
        idx = uint8(6); valid = true;
    elseif bitand(req, uint8(128)) ~= 0
        idx = uint8(7); valid = true;
    else
        idx = uint8(0); valid = false;
    end
end
