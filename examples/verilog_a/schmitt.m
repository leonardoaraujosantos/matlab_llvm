% Tier-4 example: Schmitt trigger with 1.0 V hysteresis around 1.0 V
% midpoint (vlow = 0.5 V, vhigh = 1.5 V) and rail-to-rail digital
% output.  Used for noise-immune buffering of slow analog edges.

ok = writeVerilogASchmitt(1.5, 0.5, 3.3, 0.0, "schmitt.va");
disp(ok);
