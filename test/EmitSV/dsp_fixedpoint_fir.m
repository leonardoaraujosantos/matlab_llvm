% DSP Toolbox Tier-6 SV regression — a 7-tap fixed-point FIR using the
% flat fi-array + persistent-shift-register pattern.  This is the HDL
% form of dsp.FIRFilter that the SV-emit lane handles today (the
% companion examples/dsp/fixedpoint_fir_hdl.m is the user-facing version).
%
% Bridging the full `dsp.FIRFilter('Numerator', b)` System Object to
% this lowered shape — so the SO surface ALSO reaches the SV lane — is
% the documented Tier-6 follow-on (docs/dsp_toolbox_roadmap.md §6.1–6.5).
% Today the function below is the canonical SV-emittable DSP filter.
T = numerictype(1, 16, 12);
y = dsp_fixedpoint_fir(fi(0.5, T));
disp(y);

function r = dsp_fixedpoint_fir(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    % cocotb: latency(1)
    h = fi([1, 2, 3, 4, 3, 2, 1], 1, 16, 0);
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 7), 1, 16, 12);
    end
    delay_line = [fi(x, 1, 16, 12), delay_line(1:6)];
    p1 = delay_line(1) * h(1);
    p2 = delay_line(2) * h(2);
    p3 = delay_line(3) * h(3);
    p4 = delay_line(4) * h(4);
    p5 = delay_line(5) * h(5);
    p6 = delay_line(6) * h(6);
    p7 = delay_line(7) * h(7);
    r = p1 + p2 + p3 + p4 + p5 + p6 + p7;
end
