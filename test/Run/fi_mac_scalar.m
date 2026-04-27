% Phase-1 fi: scalar multiply-accumulate (the gating example).
acc = fi(0, 1, 16, 8);
a = fi(0.5, 1, 16, 8);
b = fi(0.5, 1, 16, 8);
acc(:) = acc + a*b;
acc(:) = acc + a*b;
disp(acc);
