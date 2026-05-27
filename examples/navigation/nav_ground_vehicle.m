% Navigation Toolbox — "Estimate Position and Orientation of a Ground Vehicle".
% Mirrors https://www.mathworks.com/help/nav/ug/estimate-position-and-orientation-of-a-ground-vehicle.html
%
% The MathWorks example fuses a 6-axis `imuSensor` with a `gpsSensor` in an
% `insfilterNonholonomic` EKF.  We reuse the shipped `insfilterMARG` filter
% (a quaternion-orientation + position/velocity/bias EKF) with the same
% nested predict / fusegps loop — outer GPS rate, inner IMU rate — which is
% the canonical ground-vehicle dead-reckoning + GPS-correction structure.
% (`insfilterNonholonomic` with its no-slip side/vertical-velocity pseudo-
% measurements is a documented Tier-6 follow-on.)

imuFs = 100;              % IMU at 100 Hz
gpsFs = 10;               % GPS at 10 Hz
dt    = 1.0 / imuFs;
N     = 500;              % 5 s
gpsEvery = imuFs / gpsFs; % fuse GPS every 10 IMU steps

imu = imuSensor(imuFs);
gps = gpsSensor(gpsFs);
ins = insfilterMARG(imuFs);

% Ground truth: vehicle driving straight at 4 m/s along +x.
speed  = 4.0;
true_v = [speed, 0.0, 0.0];

fprintf('Ground-vehicle INS: IMU %d Hz / GPS %d Hz, %.1fs\n', imuFs, gpsFs, N*dt);

for k = 1:N
    % On a level road the accelerometer measures gravity; no rotation.
    z_imu = step(imu, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]);
    acc  = [z_imu(1), z_imu(2), z_imu(3)];
    gyro = [z_imu(4), z_imu(5), z_imu(6)];

    predict(ins, acc, gyro, dt);
    fuseaccel(ins, acc);

    if mod(k, gpsEvery) == 0
        true_p = [(k * dt) * speed, 0.0, 0.0];
        z_gps = step(gps, true_p, true_v);
        fusegps(ins, [z_gps(1), z_gps(2), z_gps(3)], [z_gps(4), z_gps(5), z_gps(6)]);
    end
end

S = ins.State;
true_x = N * dt * speed;
fprintf('Estimated position = [%.2f %.2f %.2f] m\n', S(5), S(6), S(7));
fprintf('Estimated velocity = [%.2f %.2f %.2f] m/s\n', S(8), S(9), S(10));
fprintf('Truth x            = %.2f m\n', true_x);
fprintf('Along-track error  = %.2f m\n', abs(S(5) - true_x));
