% Phase 5.4 — retime aggregators. 14 consecutive days resampled to
% weekly buckets with each of the 6 aggregators.

dates = datetime(2014, 1, 1) + days(0:13);
o = (100:113)';
h = (110:123)';
l = ( 95:108)';
c = (102:115)';
v = (1000:100:2300)';
TMW = timetable(o, h, l, c, v, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);

% Cadence: weekly. Mondays bound buckets.
wo = retime(TMW(:,'Open'),   'weekly', 'firstvalue');
wh = retime(TMW(:,'High'),   'weekly', 'max');
wl = retime(TMW(:,'Low'),    'weekly', 'min');
wc = retime(TMW(:,'Close'),  'weekly', 'lastvalue');
ws = retime(TMW(:,'Volume'), 'weekly', 'sum');
wm = retime(TMW(:,'Close'),  'weekly', 'mean');

disp(wo);
disp(wh);
disp(wl);
disp(wc);
disp(ws);
disp(wm);
