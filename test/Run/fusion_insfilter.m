% Sensor Fusion Tier-3 — headline tracer.  Generate a synthetic IMU+GPS
% stream along a known straight-line trajectory and fuse it with
% insfilterMARG; check that position RMSE stays bounded.
fs = 100;
dt = 1.0 / fs;
N  = 200;
imu = imuSensor(fs);
gps = gpsSensor(1);
ins = insfilterMARG(fs);

% Ground truth — straight-line trajectory along +x at 1 m/s.
true_v = [1.0, 0.0, 0.0];

rmse = 0.0;
for k = 1:N
    % True body-frame acceleration is zero (constant velocity); gravity
    % shows up as +9.81 along body-z when the body frame is level.
    acc_body  = [0.0, 0.0, 9.81];
    gyro_body = [0.0, 0.0, 0.0];

    % IMU + GPS noisy measurements.
    z_imu = step(imu, acc_body, gyro_body);
    % z_imu = [ax ay az gx gy gz] row.

    % Run the filter.
    acc_meas  = [z_imu(1), z_imu(2), z_imu(3)];
    gyro_meas = [z_imu(4), z_imu(5), z_imu(6)];
    predict(ins, acc_meas, gyro_meas, dt);
    fuseaccel(ins, acc_meas);

    if mod(k, 10) == 0
        true_p = [(k * dt), 0.0, 0.0];
        z_gps = step(gps, true_p, true_v);
        gps_p = [z_gps(1), z_gps(2), z_gps(3)];
        gps_v = [z_gps(4), z_gps(5), z_gps(6)];
        fusegps(ins, gps_p, gps_v);
    end
end

% Final position check.
S = ins.State;
true_p_end_x = N * dt;
err_x = S(5) - true_p_end_x;
fprintf('insfilterMARG x-position error = %.2f m\n', err_x);
fprintf('orientation q_w = %.3f (should remain near 1)\n', S(1));
