% Robotics — mobile-robot kinematics models (the MathWorks "Mobile Robot
% Kinematics Equations" page).  Construct each model and integrate a short
% trajectory with `derivative` + forward Euler.
%
% Models (state / command):
%   unicycleKinematics  : [x y theta] / [v omega]
%   differentialDrive   : [x y theta] / [v omega]
%   bicycleKinematics   : [x y theta] / [v psi]        (psi = steering angle)
%   ackermannKinematics : [x y theta psi] / [v psidot] (psi in the state)

dt = 0.1;
N  = 20;

uni = unicycleKinematics();
bic = bicycleKinematics(2.0);
ack = ackermannKinematics(2.0);

fprintf('Driving each model 2 s with a gentle left turn:\n');

% Unicycle: v = 1, omega = 0.3.
ux = 0; uy = 0; uth = 0;
for k = 1:N
    d = derivative(uni, [ux; uy; uth], [1.0; 0.3]);
    ux = ux + dt*d(1); uy = uy + dt*d(2); uth = uth + dt*d(3);
end
fprintf('  unicycle  -> (%.2f, %.2f, %.2f rad)\n', ux, uy, uth);

% Bicycle: v = 1, steering psi = 0.2 rad.
bx = 0; by = 0; bth = 0;
for k = 1:N
    d = derivative(bic, [bx; by; bth], [1.0; 0.2]);
    bx = bx + dt*d(1); by = by + dt*d(2); bth = bth + dt*d(3);
end
fprintf('  bicycle   -> (%.2f, %.2f, %.2f rad)\n', bx, by, bth);

% Ackermann: v = 1, steering rate psidot = 0.05 rad/s (psi starts at 0).
ax = 0; ay = 0; ath = 0; aps = 0;
for k = 1:N
    d = derivative(ack, [ax; ay; ath; aps], [1.0; 0.05]);
    ax = ax + dt*d(1); ay = ay + dt*d(2); ath = ath + dt*d(3); aps = aps + dt*d(4);
end
fprintf('  ackermann -> (%.2f, %.2f, %.2f rad, steer %.2f)\n', ax, ay, ath, aps);
