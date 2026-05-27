% Navigation Toolbox Tier-6 — GNSS pseudorange positioning (HEADLINE).
% Mirrors the "Estimate GNSS Receiver Position" workflow: from a satellite
% constellation and the pseudoranges to a receiver, recover the receiver
% position by iterative least-squares trilateration (solving for the 3-D
% position + the receiver clock bias).

% Satellite geometry (ECEF) — a stand-in for an almanac-driven constellation.
sats = gnssconstellation(0);
fprintf('Visible satellites: %d\n', size(sats, 1));

% A known receiver location (Stanford, CA).
truth = [37.4275, -122.1697, 30.0];

% Noiseless pseudoranges invert exactly.
pr = pseudoranges(truth, sats);
pos = receiverposition(pr, sats);
fprintf('True position      : lat %.4f  lon %.4f  alt %.1f\n', truth(1), truth(2), truth(3));
fprintf('Estimated position : lat %.4f  lon %.4f  alt %.1f\n', pos(1), pos(2), pos(3));

% Position error in metres (rough lat/lon-to-metre scaling).
dlat = (pos(1) - truth(1)) * 111320.0;
dlon = (pos(2) - truth(2)) * 111320.0 * cos(truth(1) * pi/180);
fprintf('Horizontal error   : %.3f m\n', sqrt(dlat*dlat + dlon*dlon));

% A noisy single-epoch GNSS fix from the gnssSensor model.
gps = gnssSensor();
fix = step(gps, truth, [0 0 0]);
fprintf('Noisy gnssSensor fix: lat %.4f  lon %.4f\n', fix(1), fix(2));
