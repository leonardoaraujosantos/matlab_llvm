% Minimal file I/O.
%
% Writes a small text file via fopen/fprintf/fclose, then reads it back
% line by line with fgetl and disp. Uses /tmp to avoid dirtying the
% source tree. String literals are double-quoted so they flow through
% the matlab_string runtime; fopen expects matlab_string* for both
% path and mode. fgetl returns a matlab_string which needs to be
% bound to a name before disp so StringBindings tracking routes to
% matlab_string_disp.

fid = fopen("/tmp/matlab_io_file_test.txt", "w");
fprintf(fid, "hello\n");
fprintf(fid, "%g\n", 42);
fprintf(fid, "world\n");
fclose(fid);

rid = fopen("/tmp/matlab_io_file_test.txt", "r");
s1 = fgetl(rid);
disp(s1);                    % hello
s2 = fgetl(rid);
disp(s2);                    % 42
s3 = fgetl(rid);
disp(s3);                    % world
fclose(rid);
