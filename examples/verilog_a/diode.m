% Tier-7 example: ideal diode (silicon-typical).  Is = 1e-14 A,
% Vt = kT/q = 25.85 mV at room temperature.

ok = writeVerilogADiode(1.0e-14, 0.02585, "diode.va");
disp(ok);
