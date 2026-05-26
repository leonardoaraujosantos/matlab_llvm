% Phase 5.4 — timetable constructor + property carrier. Build a 5-row
% OHLCV timetable from numeric column vectors plus a RowTimes axis,
% confirm height/width/numel/size report the right shape, and confirm
% disp(TT) renders the canonical Time + Variable header.

dates = datetime(2014, 1, 1) + days(0:4);
o = [100; 101; 99; 102; 104];
h = [102; 103; 101; 105; 106];
l = [ 99; 100; 98; 101; 103];
c = [101; 102; 100; 104; 105];
v = [1000; 1100; 900; 1200; 1300];

TMW = timetable(o, h, l, c, v, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);
disp(TMW);
fprintf('height = %.0f\n', height(TMW));
fprintf('width  = %.0f\n', width(TMW));
fprintf('numel  = %.0f\n', numel(TMW));
fprintf('size1  = %.0f\n', size(TMW, 1));
fprintf('size2  = %.0f\n', size(TMW, 2));
