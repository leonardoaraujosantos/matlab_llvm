% Tier-8: white-noise source.  PSD = 4*k*T*R for thermal noise of a
% 50 Ω resistor at 290 K ≈ 8e-19 V²/Hz.
ok = writeVerilogANoise(0, 8.0e-19, 0.0, "/tmp/_rf_writeva_noise_white.va");
disp(ok);              % 1
