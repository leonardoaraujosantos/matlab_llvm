% Phase-2 fi: sub-native WL (12 bits in an i16 storage lane).
% Saturate mode (Phase-1 default) clips the stored value to 12-bit range.
acc = fi(0, 1, 12, 8);
acc(:) = fi(0.5, 1, 12, 8) + fi(0.5, 1, 12, 8);
disp(acc);                 % 1.0 — fits in i12

% Saturating overflow: 4 + 4 = 8 stored in FL=8 is 2048, exceeds the
% i12 signed max of 2047, so it clamps to 2047/256 = 7.99609375.
sat = fi(0, 1, 12, 8);
sat(:) = fi(4, 1, 12, 8) + fi(4, 1, 12, 8);
disp(sat);
