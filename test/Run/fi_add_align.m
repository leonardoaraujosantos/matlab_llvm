% Phase-1 fi: add of two operands with the same FL, clamped via (:).
acc = fi(0, 1, 16, 8);
acc(:) = fi(1.5, 1, 16, 8) + fi(0.25, 1, 16, 8);
disp(acc);
