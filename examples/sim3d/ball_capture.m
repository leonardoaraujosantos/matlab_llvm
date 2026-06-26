% examples/sim3d/ball_capture.m
% --------------------------------------------------------------------
% Bouncing ball on a plane — 3-D animation AND data capture.
%
% Most sim3d examples only *render* (export an HTML player). This one also
% pulls the simulated trajectory back into the workspace so it can be saved
% and reused elsewhere (plotted, fitted, fed to another tool). Two capture
% paths are shown:
%
%   1. sim3d.capture(world, actor) — read the keyframe timeline the viewer
%      recorded, as an N-by-7 matrix [t, x,y,z, rx,ry,rz] (time + 6-DOF pose
%      per frame). This is the transform the animation actually played.
%   2. writematrix(M, file) — write that matrix to a CSV any spreadsheet or
%      numpy/pandas script can read. (csvwrite(file, M) is the legacy alias.)
%
% Run it interpreted, then open the HTML and the CSV:
%     matlabc -repl < ball_capture.m
%     xdg-open ball_capture.html
%     cat ball_trajectory.csv
%
% It also compiles and runs with byte-identical output:
%     matlabc -emit-cpp ball_capture.m > ball_capture.cpp
%     c++ -std=c++20 -I runtime ball_capture.cpp build/libMatlabRuntime.a -lm -o ball
%     ./ball

% ---------- Physics --------------------------------------------------
g  = 9.81;     % gravity                  [m/s^2]
dt = 0.01;     % frame period             [s]   (100 FPS)
N  = 220;      % frames (2.2 s)
e  = 0.75;     % coefficient of restitution (energy kept per bounce)
r  = 0.20;     % ball radius              [m]
z0 = 4.0;      % drop height              [m]

z  = z0;       % height of the ball centre
vz = 0;        % vertical velocity

% ---------- 3-D scene ------------------------------------------------
w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size  = [6 6 1];
ground.Color = [0.16 0.17 0.20];
w.add(ground);

ball = sim3d.Actor('ball', 'sphere');
ball.Color = [0.95 0.50 0.20];
w.add(ball);
ball.Scale = [2*r 2*r 2*r];     % a unit sphere scaled to diameter 2r

w.open();

% ---------- Simulate + record ----------------------------------------
fprintf('Bouncing ball: %d frames, dt=%.3f, e=%.2f.\n', N, dt, e);

for k = 1:N
    vz = vz - g*dt;
    z  = z + vz*dt;
    if z < r          % hit the floor: clamp and reflect with energy loss
        z  = r;
        vz = -e*vz;
    end
    ball.Translation = [0 0 z];
    w.run(dt);
end

w.close();

% ---------- Capture the trajectory back to the workspace -------------
M = sim3d.capture(w, ball);     % N x 7: [t, x,y,z, rx,ry,rz]
writematrix(M, 'ball_trajectory.csv');

% Post-process the captured data (proof it is real numbers, not a handle):
t      = M(:, 1);
height = M(:, 4);
zmin   = min(height);
fprintf('captured %g frames -> ball_trajectory.csv\n', size(M, 1));
fprintf('lowest ball-centre height: %.3f m (radius r = %.2f m)\n', zmin, r);

% ---------- Render ----------------------------------------------------
sim3d.export(w, 'ball_capture.html');
fprintf('wrote ball_capture.html\n');
