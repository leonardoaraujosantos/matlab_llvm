% Pragma path, signed 16-bit input ports. Asserts that
% `port(_, fi, signed, 16, 0)` emits `logic signed [15:0]` (with
% the `signed` keyword) and that the output port — driven through
% the body — keeps the same signedness.
function y = pragma_signed16(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    y = a + b;
end
