% readtable / readmatrix — CSV / delimited-text readers.
%
% Writes a small CSV via fopen/fprintf/fclose, then reads it back
% with readtable (returns a heterogeneous matlab_table) and
% readmatrix (returns a numeric matrix with NaN for non-numeric
% cells). Exercises:
%   - header detection from a row of textual labels
%   - per-column type inference (numeric / string / datetime)
%   - delimiter auto-detect (comma)
%   - readmatrix's NaN-fill on the string and date columns

fid = fopen("/tmp/matlab_readtable_test.csv", "w");
fprintf(fid, "id,name,score,when\n");
fprintf(fid, "1,alpha,3.5,2024-01-15\n");
fprintf(fid, "2,beta,4.2,2024-02-20\n");
fprintf(fid, "3,gamma,5.1,2024-03-25\n");
fclose(fid);

T = readtable("/tmp/matlab_readtable_test.csv");
disp(height(T));
disp(width(T));
disp(T);

M = readmatrix("/tmp/matlab_readtable_test.csv");
disp(size(M, 1));
disp(size(M, 2));
disp(M);
