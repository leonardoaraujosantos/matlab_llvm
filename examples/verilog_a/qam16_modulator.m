% Verilog-A QAM-16 modulator — generic I/Q modulator parameterized
% for a 2 GHz carrier.  The I and Q port voltages each take one of
% the 4-PAM levels {-3, -1, +1, +3} (scaled to V), giving a 16-
% symbol constellation:  V(out) = amp · (V(i)·cos(ωt) − V(q)·sin(ωt))
%
% Drive V(i) and V(q) externally from PAM-shaped symbol streams
% (e.g. via writeVerilogASource composed in the SPICE netlist, or
% from a digital-to-analog conversion path).  The same module
% covers QPSK, 8-PSK, and any other I/Q constellation; only the
% upstream PAM levels change.

fc  = 2.0e9;       % 2 GHz carrier
amp = 1.0;
ok = writeVerilogAIQMod(fc, amp, "qam16_modulator.va");
disp(ok);
