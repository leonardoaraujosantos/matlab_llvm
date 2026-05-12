% Tier-4: behavioral comparator with vth = 1.25 V, output 0-3.3 V,
% td = 1 ns, tr = 100 ps.
ok = writeVerilogAComparator(1.25, 3.3, 0.0, 1.0e-9, 1.0e-10, ...
                              "/tmp/_rf_writeva_comparator.va");
disp(ok);              % 1
