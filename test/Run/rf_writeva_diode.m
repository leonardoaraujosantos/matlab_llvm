% Tier-7: ideal diode.  Is = 1e-14 A (silicon-typical), Vt = 25.85 mV.
ok = writeVerilogADiode(1.0e-14, 0.02585, "/tmp/_rf_writeva_diode.va");
disp(ok);              % 1
