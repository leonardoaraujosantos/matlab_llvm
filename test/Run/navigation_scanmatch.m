% Navigation Tier-3 — lidarScan + matchScans + lidarSLAM accumulation.
ang = (-1.5:0.1:1.5)';
N = size(ang, 1);
r = 5 * ones(N, 1);
s1 = lidarScan(r, ang);
c = s1.Cartesian;
fprintf('cart0=(%.3f,%.3f)\n', c(1,1), c(1,2));
rel = matchScans(s1, s1);
fprintf('self-match dx=%.3f dy=%.3f dth=%.3f\n', rel(1), rel(2), rel(3));
slam = lidarSLAM(0.1, 8);
addScan(slam, s1);
addScan(slam, s1);
poses = addScan(slam, s1);
fprintf('slam poses=%.0f last=(%.2f,%.2f)\n', size(poses,1), poses(end,1), poses(end,2));
