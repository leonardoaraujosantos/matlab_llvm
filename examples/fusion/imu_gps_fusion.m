% Sensor Fusion and Tracking Toolbox — headline tracer.
% IMU + GPS inertial-navigation demo over a straight-line trajectory.
%
% Closes Tier-1 (quaternion math) + Tier-3 (imuSensor / gpsSensor /
% insfilterMARG) end-to-end against the shipped numeric kernel.  Run
% locally with:
%
%     ./build/matlabc -emit-llvm examples/fusion/imu_gps_fusion.m | \
%        clang++ ... -o demo && ./demo
%
% The simplified MARG (complementary-filter quaternion + gravity-
% compensated double-integrator + linear GPS correction) closes the
% headline; the full 16-state EKF with Joseph-form covariance updates is
% documented as a Tier-3 follow-on in docs/sensor_fusion_toolbox_roadmap.md.

fs = 100;
dt = 1.0 / fs;
N  = 400;

imu = imuSensor(fs);
gps = gpsSensor(1);
ins = insfilterMARG(fs);

% Ground truth: straight-line trajectory along +x at 2 m/s.
true_v = [2.0, 0.0, 0.0];

fprintf('IMU+GPS fusion: fs=%d Hz, %d steps (%.1fs flight)\n', fs, N, N*dt);

for k = 1:N
    acc_body  = [0.0, 0.0, 9.81];        % flat & level, gravity along +z
    gyro_body = [0.0, 0.0, 0.0];

    z_imu = step(imu, acc_body, gyro_body);
    acc_meas  = [z_imu(1), z_imu(2), z_imu(3)];
    gyro_meas = [z_imu(4), z_imu(5), z_imu(6)];

    predict(ins, acc_meas, gyro_meas, dt);
    fuseaccel(ins, acc_meas);

    if mod(k, 10) == 0
        true_p = [(k * dt) * 2.0, 0.0, 0.0];
        z_gps = step(gps, true_p, true_v);
        gps_p = [z_gps(1), z_gps(2), z_gps(3)];
        gps_v = [z_gps(4), z_gps(5), z_gps(6)];
        fusegps(ins, gps_p, gps_v);
    end
end

S = ins.State;
fprintf('Final state:\n');
fprintf('  position  = [%.2f, %.2f, %.2f] m\n', S(5), S(6), S(7));
fprintf('  velocity  = [%.2f, %.2f, %.2f] m/s\n', S(8), S(9), S(10));
fprintf('  quat      = [%.3f, %.3f, %.3f, %.3f]\n', S(1), S(2), S(3), S(4));
fprintf('  true x    = %.2f m (%.1fs * 2 m/s)\n', N * dt * 2.0, N * dt);
