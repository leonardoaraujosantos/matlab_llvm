% Tier-7 example: Pt-100 RTD (platinum resistance thermometer).
% R0 = 100 Ω at T0 = 273.15 K (0 °C); alpha = 3.85e-3 /K (IEC 60751).
% Uses Verilog-A's first-class $temperature for sensitivity at the
% simulation temperature.

ok = writeVerilogARTD(100.0, 3.85e-3, 273.15, "rtd_pt100.va");
disp(ok);
