% Navigation Tier-6 — GNSS pseudorange trilateration (noiseless -> exact).
sats = gnssconstellation(0);
fprintf('satellites=%.0f\n', size(sats, 1));
truth = [37.40, -122.10, 30.0];
pr = pseudoranges(truth, sats);
pos = receiverposition(pr, sats);
fprintf('lat=%.3f lon=%.3f\n', pos(1), pos(2));
fprintf('alt-err(m)=%.2f\n', abs(pos(3) - 30.0));
