% fpga_ddc.m — Digital Down-Converter receiver front-end (T8 HDL chain).
%
% The canonical FPGA RF receiver front-end: an IF-bearing input signal
% is mixed to baseband by a numerically-controlled oscillator, then
% decimated by a multi-section CIC.  This demo runs the chain in
% floating-point simulation; the synthesizable SystemVerilog emit with
% valid / backpressure-ready / reset ports + the cocotb SIL test sits
% as a documented HDL-emit follow-on (see docs/dsp_toolbox_roadmap.md).
%
% Composed surface:
%   T8  dsphdl.NCO          — local oscillator with Latency property
%   T1+ elementwise mixer    — `if_signal .* carrier` (no SO needed)
%   T8  dsphdl.CICDecimator — multiplier-free 4x decimation

fs = 1000;
f_if = 100;
f_baseband = 5;

% Build an IF-modulated signal: a low-frequency baseband modulation
% riding on a 100 Hz IF carrier.
n = 0:1023;
baseband = sin(2 * pi * f_baseband * n / fs);
if_signal = baseband .* cos(2 * pi * f_if * n / fs);

% Local oscillator at the IF frequency (mixes the IF carrier to DC).
lo = dsphdl.NCO('Frequency', f_if);
lo.SampleRate = fs;
lo.SamplesPerFrame = 1024;
carrier = lo();

% Mix down to baseband (I channel).
mixed = if_signal .* carrier;

% CIC decimate by 4: 1000 Hz -> 250 Hz, multiplier-free Hogenauer chain.
cic = dsphdl.CICDecimator(4);
cic.NumSections = 2;
recovered = cic(mixed);

fprintf('IF signal samples       = %d\n', numel(if_signal));
fprintf('mixed signal samples    = %d\n', numel(mixed));
fprintf('baseband samples (1/4)  = %d\n', numel(recovered));
fprintf('NCO peak amplitude      = %.3f\n', max(carrier));
fprintf('NCO latency             = %.0f cycles\n', lo.getLatency());
fprintf('CIC latency             = %.0f cycles\n', cic.getLatency());

% The decimated baseband contains the original 5 Hz modulation envelope.
% Energy at decimated DC is non-zero; sanity check.
fprintf('recovered RMS           = %.4f\n', sqrt(mean(recovered .^ 2)));
