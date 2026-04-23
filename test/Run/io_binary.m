% Binary file I/O: fread / fwrite.
%
% Write a small vector of doubles to a binary file, then read them
% back and display. fwrite on a matrix writes every element in memory
% order; fread(fid, n) reads n doubles and returns an n-by-1 column.

A = [1.5 2.5 3.5 4.5];

fid = fopen("/tmp/matlab_binary_test.bin", "w");
fwrite(fid, A);
fclose(fid);

rid = fopen("/tmp/matlab_binary_test.bin", "r");
B = fread(rid, 4);
fclose(rid);

disp(B);
disp(size(B, 1));            % 4
disp(size(B, 2));            % 1
