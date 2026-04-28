% Pragma path, unsigned 8-bit input port. Asserts the
% `unsigned` keyword in the pragma carries through to a `logic
% [7:0]` SV declaration without `signed`.
function y = pragma_unsigned8(a)
    %#codegen
    % hdl: port(a, fi, unsigned, 8, 0)
    y = a;
end
