% Tier-7: NTC thermistor.  R0 = 10 kΩ at T0 = 298.15 K, B = 3950 K.
ok = writeVerilogAThermistor(1.0e4, 3950.0, 298.15, ...
                              "/tmp/_rf_writeva_thermistor.va");
disp(ok);              % 1
