% Simplified save / load — custom binary format, NOT MATLAB .mat-
% compatible. One matrix per file, header = "MLB1" + rows + cols +
% double data. The API diverges from MATLAB because we don't have
% runtime variable-name metadata: save takes the value directly,
% load returns the matrix directly.

A = [1 2 3; 4 5 6];
save("/tmp/matlab_save_test.bin", A);

B = load("/tmp/matlab_save_test.bin");
disp(size(B, 1));            % 2
disp(size(B, 2));            % 3
disp(B(1, 1));               % 1
disp(B(2, 3));               % 6
disp(B(1, 3));               % 3
