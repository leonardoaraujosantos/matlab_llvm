% Pragma path, unsigned 4-bit input port + unsigned 4-bit
% persistent register. Asserts that `% hdl: port(_, fi, unsigned,
% 4, 0)` emits `logic [3:0]` (no `signed`) and that the persistent
% counter declared as `fi(_, 0, 4, 0)` follows the same shape.
function count = pragma_unsigned4(reset)
    %#codegen
    % hdl: port(reset, bool)
    persistent c;
    if isempty(c)
        c = fi(0, 0, 4, 0);
    end
    if reset
        c = fi(0, 0, 4, 0);
    else
        c = c + 1;
    end
    count = c;
end
