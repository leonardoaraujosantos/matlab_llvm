% Tier-5: VCO via idtmod.  Center freq 1 GHz, gain 100 MHz/V, amp 1 V.
ok = writeVerilogAVCO(1.0e9, 100.0e6, 1.0, "/tmp/_rf_writeva_vco.va");
disp(ok);              % 1
