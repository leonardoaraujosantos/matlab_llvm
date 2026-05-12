% Tier-4 example: 10 MHz sinusoidal stimulus source, 1 V amplitude.
% Useful as an analog testbench driver.

ok = writeVerilogASource(0, 1.0, 10.0e6, "sine_source.va");
disp(ok);
