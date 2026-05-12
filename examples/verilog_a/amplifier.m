% Verilog-A RF amplifier — linear gain + 1st-order bandwidth limit
% + tanh saturation in a single behavioral module.
%
% Use case: a typical IF / baseband amplifier in a receiver chain.
% Gain 20× (≈ 26 dB), 5 V rail-to-rail saturation, 10 MHz cutoff.

ok = writeVerilogAAmplifier(20.0, 5.0, 10.0e6, "amplifier.va");
disp(ok);
