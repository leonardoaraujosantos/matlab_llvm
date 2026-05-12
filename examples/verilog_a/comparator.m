% Tier-4 example: behavioral comparator with vth = 1.65 V (Vdd/2 for
% 3.3 V supply), digital-style 0..3.3 V output with 1 ns td + 100 ps
% tr.  Used in front-of-ADC quantizers and zero-crossing detectors.

ok = writeVerilogAComparator(1.65, 3.3, 0.0, 1.0e-9, 1.0e-10, ...
                              "comparator.va");
disp(ok);
