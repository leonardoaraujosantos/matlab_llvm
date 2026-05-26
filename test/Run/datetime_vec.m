% Phase 5.4 — vector datetime / duration. `datetime(scalar) +
% days(0:N)` produces a column of datetimes; subtracting a scalar
% datetime returns a duration_vec; length / numel report the row
% count. These are the gating ops for the timetable RowTimes axis.

base = datetime(2014, 1, 1);
dates = base + days(0:4);
disp(dates);
fprintf('length = %.0f\n', length(dates));
fprintf('numel  = %.0f\n', numel(dates));

% Vec - scalar -> duration_vec (in seconds, shown smart-unit).
gap = dates - base;
disp(gap);

% Closed-form duration_vec from a colon-range.
oneDayHourly = hours(0:6);
disp(oneDayHourly);
fprintf('length(hourly) = %.0f\n', length(oneDayHourly));
