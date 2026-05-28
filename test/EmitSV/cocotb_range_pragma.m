% Smoke test for the `% cocotb: range(<port>, <lo>, <hi>)` pragma — a
% per-port real-value bound on the random stimulus that the cocotb
% harness draws.  The default `fi_range(signed, wl, fl)` is the full
% legal range of the declared fi spec; `range(...)` is a deliberate
% narrowing for tests where (a) the natural range overflows mid-
% computation differently in SV (which truncates per op) and the Python
% reference (which saturates at growth-width), or (b) coverage-driven
% testing wants to focus on a specific value window.
%
% This module is FL=0 so it has no SV-vs-Python fi-saturation divergence
% to worry about; the range pragma just constrains the random integers
% drawn from the int16 universe into [-100, 100], which is a strict
% subset of the natural range and therefore still passes cocotb's bit-
% accuracy compare.

T = numerictype(1, 16, 0);
y = cocotb_range_pragma(fi(5, T));
disp(y);

function y = cocotb_range_pragma(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 0)
    % cocotb: latency(0)
    % cocotb: range(x, -100, 100)
    y = x + fi(7, 1, 16, 0);
end
