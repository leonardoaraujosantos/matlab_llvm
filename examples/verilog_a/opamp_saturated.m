% Tier-7 example: saturated op-amp.  V(out) = vsat * tanh(gain *
% V(vp, vn) / vsat) gives a smooth saturation profile (no slope
% discontinuity at the rail), good for SPICE convergence.

ok = writeVerilogAOpAmp(1.0e5, 12.0, "opamp_saturated.va");
disp(ok);
