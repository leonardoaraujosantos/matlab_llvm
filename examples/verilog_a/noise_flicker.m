% Tier-8 example: 1/f flicker noise source.  Typical MOSFET-style
% flicker noise: power 1e-18 V²/Hz, exponent 1.0.

ok = writeVerilogANoise(1, 1.0e-18, 1.0, "noise_flicker.va");
disp(ok);
