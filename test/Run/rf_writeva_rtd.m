% Tier-7: Pt-100 RTD.  R0 = 100 Ω at T0 = 273.15 K, alpha = 3.85e-3 /K.
ok = writeVerilogARTD(100.0, 3.85e-3, 273.15, "/tmp/_rf_writeva_rtd.va");
disp(ok);              % 1
