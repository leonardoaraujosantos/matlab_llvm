% Phase-2 fi: bin / hex / dec rendering of the stored integer.
a = fi(1.5, 1, 16, 8);     % stored = 384 = 0x180 = 110000000_2
disp(bin(a));
disp(hex(a));
disp(dec(a));
