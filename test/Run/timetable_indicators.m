% Phase 5.4 — financial indicators on a timetable. movavg + macd
% over a single-column Close TT. The synthetic series goes 100..149
% (50 points) so the simple MA's lag is easy to verify by hand and
% the MACD is well-defined for the last few rows (needs >26 to
% stabilise; we use 50).

dates = datetime(2014, 1, 1) + days(0:49);
c = (100:149)';
TMW = timetable(c, 'VariableNames', {'Close'}, 'RowTimes', dates);

% Simple 5-period MA. Row 5 = mean(100..104) = 102; row 50 = mean(145..149) = 147.
sma5 = movavg(TMW, 'simple', 5);
fprintf('sma5 head:\n');
head(sma5, 6);
fprintf('sma5 last 3:\n');
% Manual last-3 inspection via direct column read.
v = sma5.Close;
fprintf('  sma5(48..50) = %.4f %.4f %.4f\n', v(48), v(49), v(50));

% Exponential 5-period MA. Row 1 = 100 (seed); a = 2/6 = 0.3333.
% Row 50 will be close to 149 - 4.5*(1-a)^something; let me trust runtime.
ema5 = movavg(TMW, 'exponential', 5);
fprintf('ema5 head:\n');
head(ema5, 6);
v = ema5.Close;
fprintf('  ema5(48..50) = %.4f %.4f %.4f\n', v(48), v(49), v(50));

% MACD: should converge to a small near-zero value on a perfectly
% linear series (fast and slow EMA both equal x in the limit).
m = macd(TMW);
fprintf('macd  height = %.0f, width = %.0f\n', height(m), width(m));
mvals = m.MACD;
svals = m.Signal;
hvals = m.Histogram;
fprintf('  MACD(50)  = %.4f\n', mvals(50));
fprintf('  Signal(50)= %.4f\n', svals(50));
fprintf('  Hist(50)  = %.4f\n', hvals(50));
