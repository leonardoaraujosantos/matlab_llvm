% Tier-9: 1-D lookup table emission.  Synthesize a diode-like
% piecewise IV curve and emit a $table_model module + .tbl sidecar.

x = [-1.0; -0.5; 0.0; 0.4; 0.5; 0.6; 0.7; 0.8];
y = [-1.0e-9; -1.0e-9; 0.0; 1.0e-6; 1.0e-4; 1.0e-2; 0.5; 5.0];

ok = writeVerilogATable(x, y, "/tmp/_rf_writeva_table_iv.va");
disp(ok);              % 1
