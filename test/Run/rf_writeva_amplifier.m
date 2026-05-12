% RF amplifier composite block: gain × LPF × tanh-saturation.
% gain = 20, vsat = 5 V, BW_3dB = 10 MHz.
ok = writeVerilogAAmplifier(20.0, 5.0, 10.0e6, ...
                             "/tmp/_rf_writeva_amplifier.va");
disp(ok);              % 1
