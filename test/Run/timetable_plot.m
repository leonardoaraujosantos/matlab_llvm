% Phase 5.4 — plot(TT.Time, TT.Var, ...). The first arg is a
% datetime_vec; Lowering auto-wraps it with matlab_datetime_vec_to_mat
% so the existing matlab_mat-only plot backend gets a numeric x-axis
% (days from start). The PNG is rendered as a side-effect; this test
% confirms the dispatch returns without error and emits a file.

dates = datetime(2014, 1, 1) + days(0:29);
c = 100 + 5*sin((0:29)' * pi / 10);
TMW = timetable(c, 'VariableNames', {'Close'}, 'RowTimes', dates);

ema5 = movavg(TMW, 'exponential', 5);

% Render two lines: the raw series and the EMA.
plot(TMW.Time, TMW.Close, 'b');
hold on;
plot(ema5.Time, ema5.Close, 'r');
xlabel('Days from 2014-01-01');
ylabel('Close');
title('Synthetic close + 5-day EMA');
saveas(gcf, '/tmp/timetable_plot.png');

fprintf('rendered: /tmp/timetable_plot.png\n');
fprintf('series   length = %.0f\n', length(TMW.Close));
fprintf('ema5     length = %.0f\n', length(ema5.Close));
