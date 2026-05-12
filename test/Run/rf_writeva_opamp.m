% Tier-7: saturated op-amp.  Gain = 1e5, vsat = 12 V.
ok = writeVerilogAOpAmp(1.0e5, 12.0, "/tmp/_rf_writeva_opamp.va");
disp(ok);              % 1
