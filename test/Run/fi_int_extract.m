% Phase-2 fi: int(n) / storedInteger(n) — extract the underlying stored integer.
a = fi(1.5, 1, 16, 8);
disp(int(a));
disp(storedInteger(a));
b = fi(0.25, 0, 8, 6);   % unsigned, sub-native int8 lane
disp(int(b));
