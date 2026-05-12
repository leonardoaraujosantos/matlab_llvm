% Verilog-A AM modulator — V(out) = (1 + m·V(msg)) · cos(2π fc t)
%
% Center-tuned at 1 MHz with 60% modulation index — drop into a
% mixed-signal testbench against a `writeVerilogASource`-generated
% audio-band tone on V(msg).

fc = 1.0e6;            % 1 MHz carrier
m  = 0.6;              % 60% modulation index
ok = writeVerilogAAM(fc, m, "am_modulator.va");
disp(ok);
