% Tier-7 example: NTC thermistor.  R0 = 10 kΩ at T0 = 298.15 K (25 °C),
% B = 3950 K (typical for 10K3 thermistors).  Beta-equation model
% suitable for behavioral testbenches between -20 and +85 °C.

ok = writeVerilogAThermistor(1.0e4, 3950.0, 298.15, "thermistor_ntc.va");
disp(ok);
