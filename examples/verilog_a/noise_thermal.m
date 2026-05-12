% Tier-8 example: thermal-noise source for a 50 Ω resistor at 290 K.
% PSD = 4*k*T*R ≈ 8e-19 V²/Hz.  Drops a `white_noise` contribution
% that the simulator picks up during `.noise` analysis.

ok = writeVerilogANoise(0, 8.0e-19, 0.0, "noise_thermal.va");
disp(ok);
