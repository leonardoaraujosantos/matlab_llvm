% Reshape / layout tail: horzcat / vertcat / permute / squeeze /
% flip / fliplr / flipud / rot90.

A = [1 2; 3 4];
B = [5 6; 7 8];

disp(horzcat(A, B));         % [1 2 5 6; 3 4 7 8]
disp(vertcat(A, B));         % [1 2; 3 4; 5 6; 7 8]

disp(permute(A, [1 2]));     % identity: A
disp(permute(A, [2 1]));     % transpose

disp(fliplr(A));             % [2 1; 4 3]
disp(flipud(A));             % [3 4; 1 2]
disp(flip([1 2 3 4]));       % [4 3 2 1] — first non-singleton dim

disp(rot90(A));              % [2 4; 1 3]
