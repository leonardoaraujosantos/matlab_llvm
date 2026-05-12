% Tier-4: Schmitt trigger with hysteresis between 0.5 V and 1.5 V,
% output 0-3.3 V.
ok = writeVerilogASchmitt(1.5, 0.5, 3.3, 0.0, ...
                           "/tmp/_rf_writeva_schmitt.va");
disp(ok);              % 1
