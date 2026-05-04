% Workstream B SV — three distinct saturation widths in one
% module. Forces the per-width SV helper-function hoisting (post-
% B1) to emit three independent `sat_signed_<W>` functions in the
% same module without collision. Pre-B1 this lowers to inline
% ternary chains at the same three widths. Each multiply produces
% a 32-bit fi product; the three saturations clip at 24 / 20 / 16
% bits respectively, all of which are strictly narrower than the
% input so the clamp circuit is required (a 32-bit sat on a
% 32-bit input would lower to a no-op).
function [p, q, r] = sat_multi_width(a, b, c)
    %#codegen
    % hdl: port(a, fi, signed, 16, 12)
    % hdl: port(b, fi, signed, 16, 12)
    % hdl: port(c, fi, signed, 16, 12)
    % Sat at 24 bits.
    s24 = fi(a, 1, 16, 12) * fi(b, 1, 16, 12);
    p = fi(s24, 1, 24, 18, 'OverflowAction', 'Saturate');
    % Sat at 20 bits on a different intermediate.
    s20 = fi(a, 1, 16, 12) * fi(c, 1, 16, 12);
    q = fi(s20, 1, 20, 12, 'OverflowAction', 'Saturate');
    % Sat at 16 bits on a third intermediate.
    s16 = fi(b, 1, 16, 12) * fi(c, 1, 16, 12);
    r = fi(s16, 1, 16, 8, 'OverflowAction', 'Saturate');
end
