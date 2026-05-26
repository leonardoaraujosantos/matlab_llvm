% Phase 5.4 — synchronize + horz-cat. Reuses the retime fixtures but
% combines per-column slices back into a single weekly timetable, and
% then synchronizes a second TT (Volume) onto the same grid.

dates = datetime(2014, 1, 1) + days(0:13);
o = (100:113)';
h = (110:123)';
l = ( 95:108)';
c = (102:115)';
v = (1000:100:2300)';
TMW = timetable(o, h, l, c, v, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);

% Weekly OHLC slices.
wo = retime(TMW(:,'Open'),  'weekly', 'firstvalue');
wh = retime(TMW(:,'High'),  'weekly', 'max');
wl = retime(TMW(:,'Low'),   'weekly', 'min');
wc = retime(TMW(:,'Close'), 'weekly', 'lastvalue');

% Horizontal-concat the 4 single-column slices into one timetable.
weeklyTMW = [wo wh wl wc];
disp(weeklyTMW);
fprintf('horzcat height = %.0f, width = %.0f\n', ...
        height(weeklyTMW), width(weeklyTMW));

% synchronize: add the weekly-summed Volume column to weeklyTMW.
weeklyAll = synchronize(weeklyTMW, TMW(:,'Volume'), 'weekly', 'sum');
disp(weeklyAll);
fprintf('sync height = %.0f, width = %.0f\n', ...
        height(weeklyAll), width(weeklyAll));
