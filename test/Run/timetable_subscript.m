% Phase 5.4 — timetable subscripting + dot access.
%   TMW.<col>          -> matlab_mat (column data)
%   TMW.Time           -> matlab_datetime_vec (RowTimes)
%   TMW(:, 'colName')  -> matlab_timetable (single-column slice)
%   TMW(idx, :)        -> matlab_timetable (row-selected slice)
%   TMW.Properties.Description = 'X' -> property write

dates = datetime(2014, 1, 1) + days(0:4);
o = [100; 101;  99; 102; 104];
h = [102; 103; 101; 105; 106];
l = [ 99; 100;  98; 101; 103];
c = [101; 102; 100; 104; 105];
v = [1000;1100; 900;1200;1300];

TMW = timetable(o, h, l, c, v, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);
TMW.Properties.Description = 'Simulated stock data.';
disp(TMW);

% Dot-access: column matrix + RowTimes vec.
closeCol = TMW.Close;
disp(closeCol);
fprintf('rows = %.0f, cols = %.0f\n', size(closeCol, 1), size(closeCol, 2));

times = TMW.Time;
disp(times);

% Column subscript -> single-column timetable.
closeTT = TMW(:, 'Close');
disp(closeTT);
fprintf('closeTT height = %.0f, width = %.0f\n', ...
        height(closeTT), width(closeTT));

% Row subscript -> rows 2 and 4 (1-based).
idx = [2; 4];
sub = TMW(idx, :);
disp(sub);
fprintf('sub height = %.0f, width = %.0f\n', height(sub), width(sub));
