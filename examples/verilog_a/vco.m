% Tier-5 example: VCO with 1 GHz center frequency, 100 MHz/V tuning
% gain.  V(in) is the control voltage; V(out) is amp * sin(phase),
% where phase is integrated via idtmod() for clean 2*pi wrap.

ok = writeVerilogAVCO(1.0e9, 100.0e6, 1.0, "vco.va");
disp(ok);
