% sim3d.capture + writematrix/csvwrite parity (interpret vs compile).
%
% Regression guard for the data-capture surface: sim3d.capture(world, actor)
% hands the recorded keyframe timeline back as an N-by-7 matrix, and
% writematrix/csvwrite write a numeric matrix to CSV (previously unimplemented
% no-ops). The compiled lane has to type the capture result as a Matrix (not a
% bare void*) or `disp(M)` prints a pointer — this fixture catches that by
% diffing the disp'd matrix against the interpreter. Files go to /tmp so the
% harness (which runs in the invocation cwd) leaves no litter.
w = sim3d.World();
ball = sim3d.Actor('ball', 'sphere');
w.add(ball);
w.open();
z = 4.0; vz = 0;
for k = 1:5
    vz = vz - 9.81*0.1;
    z = z + vz*0.1;
    ball.Translation = [0 0 z];
    w.run(0.1);
end
w.close();

M = sim3d.capture(w, ball);     % N x 7: [t, x,y,z, rx,ry,rz]
disp('capture:'); disp(M);
disp('size:'); disp(size(M));

rc = writematrix(M, '/tmp/sim3d_capture_diff.csv');
disp('writematrix rc:'); disp(rc);
csvwrite('/tmp/sim3d_capture_diff2.csv', M);

% Read the first row back to prove the file was actually written (and that the
% two lanes wrote the same bytes).
fid = fopen('/tmp/sim3d_capture_diff.csv', 'r');
line = fgetl(fid);
fclose(fid);
disp('first row:'); disp(line);
