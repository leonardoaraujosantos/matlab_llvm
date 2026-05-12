% Tier-8: 1/f flicker noise.  pwr = 1e-18, exponent = 1.0 (pure 1/f).
ok = writeVerilogANoise(1, 1.0e-18, 1.0, ...
                         "/tmp/_rf_writeva_noise_flicker.va");
disp(ok);              % 1
