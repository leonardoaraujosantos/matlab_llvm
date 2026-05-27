% Navigation Tier-5 — controllerVFH reactive steering (deterministic, no RNG).
vfh = controllerVFH();
ang = (-pi/2:0.1:pi/2)';
N = size(ang, 1);
% Open field, target straight ahead -> steer stays near 0.
r2 = 10 * ones(N, 1);
steer2 = step(vfh, r2, ang, 0.0);
fprintf('open-field steer = %.2f\n', steer2);
% Obstacle dead-ahead with the target biased right -> steer turns right (+).
r = 10 * ones(N, 1);
for k = 1:N
    if abs(ang(k)) < 0.35
        r(k) = 0.6;
    end
end
steer = step(vfh, r, ang, 0.6);
fprintf('blocked-ahead steer = %.2f\n', steer);
