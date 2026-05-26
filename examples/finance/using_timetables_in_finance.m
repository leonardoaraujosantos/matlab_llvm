% using_timetables_in_finance.m — port of MathWorks
% "Using Timetables in Finance" (Financial Toolbox doc-page) onto
% matlab_llvm. Synthetic OHLCV in place of the SimulatedStock.mat
% fixture (no MAT-file decoder yet); everything else mirrors the
% canonical workflow:
%
%   1. construct a timetable + set Properties.Description
%   2. summary / fillmissing (NaN repair)
%   3. timerange row subscript + head
%   4. column subscript -> EMA + MACD
%   5. weekly retime + horz-cat + synchronize
%   6. plot with datetime x-axis

rng(0);

n = 60;
dates = datetime(2014, 1, 1) + days(0:(n-1));
close = 100 + cumsum(randn(n, 1) * 0.5);
open  = close + randn(n, 1) * 0.2;
high  = max(open, close) + abs(randn(n, 1)) * 0.3;
low   = min(open, close) - abs(randn(n, 1)) * 0.3;
vol   = 1000 + abs(randn(n, 1)) * 200;

% Sprinkle 3 NaNs into Close (rows 15, 30, 45).
close(15) = NaN;
close(30) = NaN;
close(45) = NaN;

TMW = timetable(open, high, low, close, vol, ...
                'VariableNames', {'Open','High','Low','Close','Volume'}, ...
                'RowTimes', dates);
TMW.Properties.Description = 'Simulated stock data.';

fprintf('--- summary (raw) ---\n');
summary(TMW);

TMW = fillmissing(TMW, 'linear');
fprintf('--- summary (filled) ---\n');
summary(TMW);

% Time-range subscript: ~ a month in mid-series.
idx = timerange(datetime(2014, 1, 15), datetime(2014, 2, 15), 'closed');
fprintf('--- TMW(timerange, :) head ---\n');
head(TMW(idx, :), 4);

% Column subscript -> 15-period EMA + MACD on Close.
closeTT = TMW(:, 'Close');
ema15 = movavg(closeTT, 'exponential', 15);
fprintf('--- ema15 head ---\n');
head(ema15, 6);

mline = macd(closeTT);
fprintf('macd shape: height=%.0f, width=%.0f\n', height(mline), width(mline));

% Weekly aggregation: 4 single-column TTs -> horz-cat -> synchronize.
wo = retime(TMW(:, 'Open'),   'weekly', 'firstvalue');
wh = retime(TMW(:, 'High'),   'weekly', 'max');
wl = retime(TMW(:, 'Low'),    'weekly', 'min');
wc = retime(TMW(:, 'Close'),  'weekly', 'lastvalue');
weeklyTMW = [wo wh wl wc];
weeklyAll = synchronize(weeklyTMW, TMW(:, 'Volume'), 'weekly', 'sum');
fprintf('--- weeklyAll ---\n');
disp(weeklyAll);
fprintf('weeks: %.0f, vars: %.0f\n', height(weeklyAll), width(weeklyAll));

% Plot the close + EMA overlay on a datetime x-axis.
plot(TMW.Time, TMW.Close, 'b');
hold on;
plot(ema15.Time, ema15.Close, 'r');
xlabel('Days from 2014-01-01');
ylabel('Close');
title('Synthetic stock close + 15-day EMA');
saveas(gcf, '/tmp/using_timetables_in_finance.png');
fprintf('rendered: /tmp/using_timetables_in_finance.png\n');
