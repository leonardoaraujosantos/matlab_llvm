% I/Q modulator.  Carrier 2 GHz, output amplitude 1 V.
ok = writeVerilogAIQMod(2.0e9, 1.0, "/tmp/_rf_writeva_iqmod.va");
disp(ok);              % 1
