% Tier-6 example: pure-Verilog-A 8-bit DAC.  vref = 3.3 V, 1 ns delay
% + 100 ps rise time per code transition.  V(code) is read as an
% analog-coded voltage in [0, 255] -> V(out) in [0, vref].

ok = writeVerilogADAC(8, 3.3, 1.0e-9, 1.0e-10, "dac_8bit.va");
disp(ok);
