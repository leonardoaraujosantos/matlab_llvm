% Robotics Tier-5 follow-on — unicycle / bicycle / ackermann kinematics
% (the "Mobile Robot Kinematics Equations" models) + derivative.
uni = unicycleKinematics();
bic = bicycleKinematics(2.0);     % 2 m wheelbase
ack = ackermannKinematics(2.0);

% Unicycle: state [x y theta], cmd [v omega]; thetadot = omega.
du = derivative(uni, [0.0; 0.0; 0.0], [1.0; 0.5]);
fprintf('unicycle ddot = [%.3f %.3f %.3f]\n', du(1), du(2), du(3));

% Bicycle: cmd [v psi]; thetadot = v*tan(psi)/L.  v=2, psi=pi/4, L=2 -> 1.0.
db = derivative(bic, [0.0; 0.0; 0.0], [2.0; pi/4]);
fprintf('bicycle ddot = [%.3f %.3f %.3f]\n', db(1), db(2), db(3));

% Ackermann: state [x y theta psi], cmd [v psidot]; psidot passes through.
da = derivative(ack, [0.0; 0.0; 0.0; pi/4], [2.0; 0.3]);
fprintf('ackermann ddot = [%.3f %.3f %.3f %.3f]\n', da(1), da(2), da(3), da(4));

% Euler-integrate the bicycle 10 steps at dt=0.1 with constant [v psi].
x = 0.0; y = 0.0; th = 0.0;
for k = 1:10
    d = derivative(bic, [x; y; th], [1.0; 0.2]);
    x  = x  + 0.1 * d(1);
    y  = y  + 0.1 * d(2);
    th = th + 0.1 * d(3);
end
fprintf('bicycle pose after 1s: (%.3f, %.3f, %.3f)\n', x, y, th);
