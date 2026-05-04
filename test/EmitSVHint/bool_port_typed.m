% D1 lint hint — control case. Reset is declared as `bool` so
% the SV port is already 1-bit. The emitter should NOT emit any
% port-width hint here; the runner asserts stderr is clean.
function y = bool_port_typed(reset)
    %#codegen
    % hdl: port(reset, bool)
    if reset
        y = fi(0, 1, 16, 0);
    else
        y = fi(1, 1, 16, 0);
    end
end
