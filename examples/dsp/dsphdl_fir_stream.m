% dsphdl_fir_stream.m — DSP HDL Toolbox Tier-7 simulation surface.
%
% `dsphdl.FIRFilter` is the cycle-accurate hardware counterpart of
% `dsp.FIRFilter`.  In this MATLAB-side simulation it computes the same
% reference result as the dsp.* sibling; the synthesizable SystemVerilog
% emit with valid / backpressure-ready / reset ports plus the cocotb SIL
% are the next slice (the emit-SV lane needs new patterns for clocked
% valid/ready datapaths — a follow-on roadmap item).

% Design a smoothing FIR.
b = fir1(15, 0.25);

% Identical taps drive both the dsp.* and dsphdl.* objects.
sim = dsp.FIRFilter('Numerator',  b);
hw  = dsphdl.FIRFilter('Numerator', b);

% A multi-frame streaming run.  Each frame carries state forward via the
% handle classdef; the HW object will, in the future, do the same on the
% RTL side with valid-gated state updates.
n = 0:255;
x = sin(2 * pi * 0.05 * n) + 0.4 * randn(1, 256);
y_sim = zeros(1, 256);
y_hw  = zeros(1, 256);
for k = 1:8
    idx = (k - 1) * 32 + (1:32);
    y_sim(idx) = sim(x(idx));
    y_hw(idx)  = hw(x(idx));
end

fprintf('sim vs hw maxdiff = %.6f\n', max(abs(y_sim - y_hw)));
fprintf('hw FIR latency    = %.0f clock cycles\n', hw.getLatency());

% CIC + NCO front-end (the canonical digital-down-converter shape).
nco = dsphdl.NCO('Frequency', 50);
nco.SampleRate = 1000;
nco.SamplesPerFrame = 128;
cic = dsphdl.CICDecimator(4);
cic.NumSections = 2;

mix = nco();
xdec = cic(mix);
fprintf('NCO/CIC chain: in=%d out=%d\n', numel(mix), numel(xdec));
fprintf('NCO latency = %.0f\n', nco.getLatency());
fprintf('CIC latency = %.0f\n', cic.getLatency());

% CORDIC math: atan2 + sqrt on a vector (function form, no SO needed).
y_ax = [0 1 1  0];
x_ax = [1 1 0 -1];
ang  = cordic_atan2(y_ax, x_ax);
fprintf('atan2 angles: %.2f %.2f %.2f %.2f\n', ang(1), ang(2), ang(3), ang(4));
