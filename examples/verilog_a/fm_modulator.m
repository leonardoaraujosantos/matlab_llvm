% Verilog-A FM modulator — uses the same writeVerilogAVCO entry
% that powers the VCO example, in FM-modulator framing.
%
% f(t) = fc + kf · V(msg);  phase = idtmod(2π f(t), 0, 2π);
% V(out) = amp · sin(phase).
%
% Center 100 MHz carrier, 10 MHz/V frequency deviation gain
% (so ±1 V on msg drives ±10 MHz frequency swing — wideband FM).

fc  = 100.0e6;         % 100 MHz center
kf  = 10.0e6;          % 10 MHz/V deviation gain
amp = 1.0;
ok = writeVerilogAVCO(fc, kf, amp, "fm_modulator.va");
disp(ok);
