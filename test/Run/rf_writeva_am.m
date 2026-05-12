% AM modulator.  Carrier 1 MHz, modulation index 0.6.
ok = writeVerilogAAM(1.0e6, 0.6, "/tmp/_rf_writeva_am.va");
disp(ok);              % 1
