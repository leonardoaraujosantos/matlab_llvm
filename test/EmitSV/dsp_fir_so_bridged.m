% DSP toolbox SO -> SV bridge — same algorithm shape as
% test/EmitSV/dsp_fixedpoint_fir.m, but the user writes
% `dsp.FIRFilter('Numerator', b)` instead of the flat fi-array
% shift-register loop.  The LowerDspSystemObjects rewrite recognises
% the SO construction + step pattern and substitutes the flat form
% before the rest of the SV pipeline runs.
T = numerictype(1, 16, 12);
y = dsp_fir_so_bridged(fi(0.5, T));
disp(y);

function r = dsp_fir_so_bridged(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    % cocotb: latency(1)
    persistent firFilt;
    if isempty(firFilt)
        firFilt = dsp.FIRFilter('Numerator', fi([1 2 3 4 3 2 1], 1, 16, 0));
    end
    r = firFilt(fi(x, 1, 16, 12));
end
