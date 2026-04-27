% Phase 5.6.1 SV — `% hdl: port(...)` pragma fixes port widths
% on a function-only .m file (no typed driver). Mirrors the
% bundled-driver `add_scalar.m` fixture.
function y = port_pragma(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(y, fi, signed, 16, 0)
    y = a + b;
end
