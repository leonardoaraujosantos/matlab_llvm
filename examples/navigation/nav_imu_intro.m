% Navigation Toolbox — "Introduction to Simulating IMU Measurements".
% Mirrors https://www.mathworks.com/help/nav/ug/introduction-to-simulating-imu-measurements.html
%
% Navigation reuses the Sensor Fusion `imuSensor` model.  An IMU mounted on
% a platform reports specific force (accelerometer) and angular velocity
% (gyroscope).  Here we drive the model with a known motion profile — a
% level platform turning at a constant yaw rate — and read back the
% simulated, noise-corrupted measurements that a navigation filter would
% consume.

fs = 100;                 % 100 Hz IMU
dt = 1.0 / fs;
N  = 200;                 % 2 seconds

imu = imuSensor(fs);

% Motion profile: flat & level (gravity reads on +Z), yawing at 0.5 rad/s.
acc_body  = [0.0, 0.0, 9.81];
gyro_body = [0.0, 0.0, 0.5];

fprintf('Simulating IMU: %d Hz, %d samples (%.1fs)\n', fs, N, N*dt);

ax_sum = 0.0; gz_sum = 0.0;
for k = 1:N
    z = step(imu, acc_body, gyro_body);
    ax_sum = ax_sum + z(1);
    gz_sum = gz_sum + z(6);
    if k <= 3
        fprintf('  k=%d  accel=[%.3f %.3f %.3f]  gyro=[%.3f %.3f %.3f]\n', ...
                k, z(1), z(2), z(3), z(4), z(5), z(6));
    end
end

fprintf('mean accel-x = %.4f m/s^2 (truth 0)\n', ax_sum / N);
fprintf('mean gyro-z  = %.4f rad/s  (truth 0.5)\n', gz_sum / N);
