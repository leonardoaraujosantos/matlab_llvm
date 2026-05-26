% Financial Toolbox Tier-1 §1 — date arithmetic
d1 = datetime(2014, 1, 1);
d2 = datetime(2014, 7, 1);     % 6 months later
d3 = datetime(2015, 1, 1);     % 1 year later

fprintf('yearfrac act/act = %.4f\n', yearfrac(d1, d3, 0));   % ~1.0
fprintf('yearfrac 30/360  = %.4f\n', yearfrac(d1, d2, 1));   % 0.5
fprintf('yearfrac act/360 = %.4f\n', yearfrac(d1, d3, 2));   % ~1.0139
fprintf('yearfrac act/365 = %.4f\n', yearfrac(d1, d3, 3));   % ~1.0

fprintf('daysdif act    = %.0f\n', daysdif(d1, d3, 3));        % 365
fprintf('daysdif 30/360 = %.0f\n', daysdif(d1, d3, 1));        % 360
fprintf('days360        = %.0f\n', days360(d1, d3));           % 360
fprintf('days365        = %.0f\n', days365(d1, d3));           % 365

% daysadd: add 30 days.
d4 = daysadd(d1, 30, 1);
disp(d4);                                                       % 31-Jan-2014 00:00:00

% isbusday: 1 Jan 2014 is a Wednesday -> 1.
fprintf('isbusday Wed = %.0f\n', isbusday(d1));
% 4 Jan 2014 is a Saturday -> 0.
fprintf('isbusday Sat = %.0f\n', isbusday(datetime(2014, 1, 4)));

% busdate forward from Friday Jan 3 -> Monday Jan 6.
d5 = busdate(datetime(2014, 1, 3), 1);
disp(d5);

% eomdate(2014, 2) -> 28-Feb-2014.
eom = eomdate(2014, 2);
disp(eom);

% Excel <-> MATLAB date number conversion (round-trip).
mn = 735600;            % MATLAB date for ~Jan 1, 2014
xd = m2xdate(mn);
fprintf('xdate = %.0f\n', xd);
fprintf('round-trip = %.0f\n', x2mdate(xd));
