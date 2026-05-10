% CSV / delimited-text readers — readmatrix and readtable.
%
% Two passes over /tmp:
%   1. A pure-numeric CSV read with readmatrix into a double matrix.
%   2. A heterogeneous CSV (numbers, strings, dates) read with
%      readtable, which infers a kind per column.
%
% The CSV files are written inline with fprintf so the example is
% self-contained. The delimiter (',') and the header row are auto-
% detected; the type of each column is inferred from its cells.

% ---- 1. Numeric-only CSV --------------------------------------------
%
% Three variables (a, b, c) sampled at four time points. readmatrix
% returns a plain double matrix; we reduce it with sum/mean and slice
% out a single column.

fid = fopen("/tmp/matlab_csv_numeric.csv", "w");
fprintf(fid, "a,b,c\n");
fprintf(fid, "1,10,100\n");
fprintf(fid, "2,20,200\n");
fprintf(fid, "3,30,300\n");
fprintf(fid, "4,40,400\n");
fclose(fid);

M = readmatrix("/tmp/matlab_csv_numeric.csv");
disp("M =");
disp(M);

% Per-column sum: sum(M) reduces along dim 1 and returns a row vector.
disp("sum(M)  =");
disp(sum(M));

% Per-column mean.
disp("mean(M) =");
disp(mean(M));

% Slice out the second column (variable b) — column index 2, all rows.
disp("M(:, 2) =");
disp(M(:, 2));


% ---- 2. Heterogeneous CSV (numbers + strings + dates) ---------------
%
% A small ledger of transactions. readtable picks NUMERIC for `id` and
% `amount`, STRING for `who`, DATETIME for `when`. The shape queries
% (height/width) operate on the table descriptor; numeric columns can
% be reduced like any other matrix.

fid = fopen("/tmp/matlab_csv_mixed.csv", "w");
fprintf(fid, "id,who,amount,when\n");
fprintf(fid, "1,alice,12.50,2025-01-10\n");
fprintf(fid, "2,bob,7.25,2025-01-12\n");
fprintf(fid, "3,carol,42.00,2025-02-03\n");
fprintf(fid, "4,dave,19.75,2025-02-14\n");
fclose(fid);

T = readtable("/tmp/matlab_csv_mixed.csv");

disp("height(T) =");
disp(height(T));
disp("width(T)  =");
disp(width(T));

disp("T =");
disp(T);

% Numeric columns flow through the normal matrix path.
amount = T.amount;
disp("total amount =");
disp(sum(amount));
disp("mean amount  =");
disp(mean(amount));
