% Tier-4: behavioral sinusoidal source.  1 MHz, 1 V amplitude.
ok = writeVerilogASource(0, 1.0, 1.0e6, "/tmp/_rf_writeva_src_sine.va");
disp(ok);              % 1
