% plot_trajectory.m — render the 3-D flight path logged by a quadrotor
% mflowLink model's `signal_scope3d` block.
%
% Two-step usage (the simulator writes a CSV, this script plots it):
%
%   build/matlabc -simulate examples/quadrotor/mflowlink/quadrotor_mpc.mflow \
%       > quadrotor_traj.csv
%   build/matlabc -repl examples/quadrotor/mflowlink/plot_trajectory.m
%
% The CSV columns are `t, traj[x], traj[y], traj[z]` (the scope3d group), so
% the trajectory is columns 2:4. Renders the (x,y,z) path with start/end
% markers and the commanded setpoint, and saves it to quadrotor_traj.png.

% Prerequisite: generate quadrotor_traj.csv first (see the two-step usage
% above). Needs a plotting-enabled build (-DMATLAB_LLVM_WITH_PLOT=ON).
% (A `if exist(csvfile,'file')` guard would be nicer, but `exist` is not yet
%  implemented — tracked in issue #404; `error()`/try-catch gaps in #405.)
csvfile = 'quadrotor_traj.csv';
M = readmatrix(csvfile);          % columns: t, traj[x], traj[y], traj[z]

x = M(:, 2);
y = M(:, 3);
z = M(:, 4);

figure;
plot3(x, y, z);                   % the flown path
hold on;
plot3(x(1),   y(1),   z(1),   'go');   % start  (origin)
plot3(x(end), y(end), z(end), 'rs');   % settle point
plot3(1.0, 1.5, 1.0, 'kx');            % commanded setpoint (x=1, y=1.5, z=1)
grid on;
title('Quadrotor 3-D flight path (signal\_scope3d)');
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
legend('path', 'start', 'end', 'setpoint');

saveas(gcf, 'quadrotor_traj.png');
fprintf('Saved 3-D trajectory to quadrotor_traj.png (%d samples)\n', numel(x));
