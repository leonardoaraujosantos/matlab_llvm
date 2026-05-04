% Workstream B SV — saturation feeding directly into a function
% return value (no persist set-site, no intermediate alloca slot).
% Verifies the per-module sat helper (post-B1) renders uniformly
% whether the saturating SelectOp lands at a register `_next`
% assignment or at a `func.return` operand.
function r = sat_in_return(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 8)
    % hdl: port(b, fi, signed, 16, 8)
    s = fi(a, 1, 16, 8) * fi(b, 1, 16, 8);
    % Saturate the wide product into 16-bit signed and return.
    r = fi(s, 1, 16, 8, 'OverflowAction', 'Saturate');
end
