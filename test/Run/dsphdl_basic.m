% DSP HDL Toolbox Tier-7/8 — simulation surface + CORDIC math.
%
% The dsphdl.* System Objects compute the same reference result as their
% dsp.* siblings in floating-point simulation, with an added Latency
% property and getLatency method matching the MathWorks `dsphdl.*` API.
% The synthesizable SystemVerilog emit + cocotb SIL is a documented
% follow-on (requires new emit-SV lane patterns for clocked valid/ready
% datapaths); what ships here is the script-level API.

b = [0.25 0.5 0.25];
ref = dsp.FIRFilter('Numerator', b);
hw  = dsphdl.FIRFilter('Numerator', b);
x = [1 0 0 0 0];
y_sim = ref(x);
y_hw  = hw(x);
fprintf('hw vs sim maxdiff = %.6f\n', max(abs(y_hw - y_sim)));
fprintf('hw FIR latency    = %.0f\n', hw.getLatency());

nco = dsphdl.NCO('Frequency', 100);
nco.SampleRate = 1000;
nco.SamplesPerFrame = 50;
y = nco();
fprintf('hw NCO peak       = %.3f  latency = %.0f\n', max(y), nco.getLatency());

cic = dsphdl.CICDecimator(4);
cic.NumSections = 2;
xc = sin(2 * pi * 0.02 * (0:127));
yc = cic(xc);
fprintf('hw CIC out len    = %d   latency = %.0f\n', numel(yc), cic.getLatency());

% Function-form CORDIC math (Tier-8): vectorised element-wise.
ang = cordic_atan2([0 1 0 -1], [1 0 -1 0]);
fprintf('cordic atan2: %.3f %.3f %.3f %.3f\n', ang(1), ang(2), ang(3), ang(4));
sq = cordic_sqrt([0 1 4 9 16]);
fprintf('cordic sqrt:  %.0f %.0f %.0f %.0f %.0f\n', ...
        sq(1), sq(2), sq(3), sq(4), sq(5));
