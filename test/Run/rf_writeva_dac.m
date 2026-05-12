% Tier-6: behavioral 8-bit DAC, vref = 3.3 V, td = 1 ns, tr = 100 ps.
ok = writeVerilogADAC(8, 3.3, 1.0e-9, 1.0e-10, ...
                       "/tmp/_rf_writeva_dac.va");
disp(ok);              % 1
