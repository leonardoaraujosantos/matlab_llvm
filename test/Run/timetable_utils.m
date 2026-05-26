% Phase 5.4 — timetable utilities. fillmissing (linear) repairs NaN
% holes; summary prints per-column stats with NumMissing; head shows
% the first n rows.

dates = datetime(2014, 1, 1) + days(0:9);
% Close has 3 NaNs at rows 3, 5, 8 (1-based: rows 4, 6, 9).
c = [101; 102; 103; NaN; 105; NaN; 107; 108; NaN; 110];
v = (1000:100:1900)';
TMW = timetable(c, v, 'VariableNames', {'Close','Volume'}, 'RowTimes', dates);

% Summary should report NumMissing=3 for Close, 0 for Volume.
summary(TMW);

% Linear fill: row 4 = 104, row 6 = 106, row 9 = 109 (linear neighbours).
filled = fillmissing(TMW, 'linear');
summary(filled);
disp(filled);

% head(TT, n) limits to first n rows.
fprintf('--- head(filled, 4) ---\n');
head(filled, 4);
