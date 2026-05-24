% fixedpoint_fir_hdl.m — DSP Toolbox Tier-6 (HDL track).
%
% This is the FIR you reach for when the target is real silicon: a
% fixed-point implementation with a persistent tapped-delay line and a
% constant-coefficient table.  The same algorithm shape is exposed by
% `dsp.FIRFilter` (Tier-1, floating-point reference / streaming sim);
% here we use the flat function form because it is the form that lowers
% to synthesizable SystemVerilog today via the shipped persistent-fi ->
% SV regfile lane.
%
% Run it three ways:
%
%   1. Float reference (script-side caller above):
%      ./build/matlabc -emit-llvm examples/dsp/fixedpoint_fir_hdl.m | \
%        clang++ - <link-runtime> && ./a.out
%
%   2. Hardware emit (the headline):
%      ./build/matlabc -emit-systemverilog examples/dsp/fixedpoint_fir_hdl.m
%
%   3. Synthesizability gate:
%      ./build/matlabc -check-synthesizable examples/dsp/fixedpoint_fir_hdl.m
%
% Bridging this flat function back to `dsp.FIRFilter('Numerator', b)`
% with a fi-typed Numerator + DiscreteState — so the SO surface ALSO
% reaches the SV lane — is the documented Tier-6 follow-on (see
% docs/dsp_toolbox_roadmap.md §6.1–6.5).

% Primary use: drive the SV emit on the function below.
%   matlabc -emit-systemverilog examples/dsp/fixedpoint_fir_hdl.m
% The script body here is the floating-point reference driver used by
% the SV-emit test harness (test/EmitSV/fir_filter.m precedent).  The
% persistent fi-array slice store and `double(fi)` cast both lower in
% the SV-emit pipeline; this script body works in that mode.
T = numerictype(1, 16, 12);
y = fir_filter_fi(fi(0.5, T));
disp(y);

% Synthesizable 7-tap fixed-point FIR.  The `hdl: port` directives
% pin the I/O fi types; the runtime emits a clocked SV module with
% the persistent delay line as N parallel registers and the
% constant coefficient table as a static SV lookup.
function r = fir_filter_fi(x)
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
