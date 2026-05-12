% Tier-9 example: 1-D lookup table.  Synthesizes a diode-like
% piecewise IV curve and emits a $table_model module + .tbl sidecar.
% The .tbl is loaded at simulation time and interpolated linearly.

x = [-1.0; -0.5; 0.0; 0.4; 0.5; 0.6; 0.7; 0.8];
y = [-1.0e-9; -1.0e-9; 0.0; 1.0e-6; 1.0e-4; 1.0e-2; 0.5; 5.0];

ok = writeVerilogATable(x, y, "iv_curve.va");
disp(ok);
