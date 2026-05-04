% D1 lint hint — input port declared as 8-bit fi but only used in
% `!= 0` boolean predicates. The emitter should warn and suggest
% retyping the source to `bool` or 1-bit fi so the SV port is the
% expected single bit. The module still emits valid synthesizable
% RTL; this is informational, not a hard error.
function y = bool_port_8bit(reset)
    %#codegen
    % hdl: port(reset, fi, unsigned, 8, 0)
    if reset
        y = fi(0, 1, 16, 0);
    else
        y = fi(1, 1, 16, 0);
    end
end
