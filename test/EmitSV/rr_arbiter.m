function [grant, valid] = rr_arbiter(req, reset)
    %#codegen
    % hdl: port(req, fi, unsigned, 8, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 4-input round-robin arbiter. `req` carries 4 request lines in
    % its low nibble; `grant` returns the index (0..3) of the
    % granted request, `valid` is high when at least one request
    % was outstanding. After each grant the priority pointer
    % advances by 1, so the next cycle prefers the next requestor.
    % If no request is outstanding the pointer holds.
    %
    % Real arbiters use a barrel-shifted priority chain; this
    % unrolled form synthesizes to the same priority mux. Tests:
    %   - persistent priority pointer (mod-4 counter advanced
    %     conditionally on grant)
    %   - 4 nested if-priority decisions per starting position
    %   - dual-output design where both outputs depend on the
    %     pointer's current snapshot
    %
    % Storage layout: bit K of `req` is request from requestor K.

    persistent prio;
    if isempty(prio) || reset
        prio = uint8(0);
    end

    p = prio + uint8(0);
    valid = req ~= 0;
    grant = uint8(0);

    % Compute the grant by checking each requestor in priority
    % order, starting from `prio`. The 4 inner branches per
    % `prio` value are unrolled — the synth tool collapses
    % redundant terms into one mux.
    if p == 0
        if bitand(req, uint8(1)) ~= 0;     grant = uint8(0);
        elseif bitand(req, uint8(2)) ~= 0; grant = uint8(1);
        elseif bitand(req, uint8(4)) ~= 0; grant = uint8(2);
        elseif bitand(req, uint8(8)) ~= 0; grant = uint8(3);
        end
    elseif p == 1
        if bitand(req, uint8(2)) ~= 0;     grant = uint8(1);
        elseif bitand(req, uint8(4)) ~= 0; grant = uint8(2);
        elseif bitand(req, uint8(8)) ~= 0; grant = uint8(3);
        elseif bitand(req, uint8(1)) ~= 0; grant = uint8(0);
        end
    elseif p == 2
        if bitand(req, uint8(4)) ~= 0;     grant = uint8(2);
        elseif bitand(req, uint8(8)) ~= 0; grant = uint8(3);
        elseif bitand(req, uint8(1)) ~= 0; grant = uint8(0);
        elseif bitand(req, uint8(2)) ~= 0; grant = uint8(1);
        end
    else
        if bitand(req, uint8(8)) ~= 0;     grant = uint8(3);
        elseif bitand(req, uint8(1)) ~= 0; grant = uint8(0);
        elseif bitand(req, uint8(2)) ~= 0; grant = uint8(1);
        elseif bitand(req, uint8(4)) ~= 0; grant = uint8(2);
        end
    end

    % Advance pointer on a successful grant; wrap mod-4.
    if valid
        if p == 3
            prio = uint8(0);
        else
            prio = p + uint8(1);
        end
    end
end
