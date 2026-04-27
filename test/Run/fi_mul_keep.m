% Phase-1 fi: scalar multiply, clamped back to the input spec via (:).
acc = fi(0, 1, 16, 8);
acc(:) = fi(1.5, 1, 16, 8) * fi(0.5, 1, 16, 8);
disp(acc);
