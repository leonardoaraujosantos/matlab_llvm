% Phase 5.4 — timerange + TT(timerange, :) row subscript. A timerange
% object represents [t1, t2] (or half-open variants) and selects
% timetable rows whose RowTime falls in the interval.

dates = datetime(2014, 1, 1) + days(0:9);
c = (101:110)';
v = (1000:100:1900)';
TMW = timetable(c, v, 'VariableNames', {'Close','Volume'}, 'RowTimes', dates);

% Closed range [Jan 3, Jan 6] -> 4 rows.
tr = timerange(datetime(2014,1,3), datetime(2014,1,6), 'closed');
sub = TMW(tr, :);
disp(sub);
fprintf('closed   height = %.0f\n', height(sub));

% Open-right [Jan 3, Jan 6) -> 3 rows.
tr2 = timerange(datetime(2014,1,3), datetime(2014,1,6), 'openright');
sub2 = TMW(tr2, :);
fprintf('openright height = %.0f\n', height(sub2));

% Open-left (Jan 3, Jan 6] -> 3 rows.
tr3 = timerange(datetime(2014,1,3), datetime(2014,1,6), 'openleft');
sub3 = TMW(tr3, :);
fprintf('openleft  height = %.0f\n', height(sub3));

% Fully open (Jan 3, Jan 6) -> 2 rows.
tr4 = timerange(datetime(2014,1,3), datetime(2014,1,6), 'open');
sub4 = TMW(tr4, :);
fprintf('open      height = %.0f\n', height(sub4));
