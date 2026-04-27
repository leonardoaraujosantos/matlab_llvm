% Phase-1 fi: unsigned scalar arithmetic.
acc = fi(0, 0, 16, 8);
acc(:) = fi(2.5, 0, 16, 8) + fi(1.5, 0, 16, 8);
disp(acc);
