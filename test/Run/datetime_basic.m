% Phase 5.1 — datetime / duration. Constructors, display, and basic
% arithmetic (datetime ± duration, datetime - datetime, duration ±
% duration). The datetime descriptor stores Unix-epoch seconds; the
% duration descriptor stores a relative span in seconds.

t = datetime(2026, 5, 1, 12, 30, 0);
disp(t);                       % 01-May-2026 12:30:00

t0 = datetime(2026, 5, 1);
disp(t0);                      % 01-May-2026 00:00:00

% Duration constructors with smart-unit display.
s = seconds(45);
disp(s);                       % 45.000000 sec
m = minutes(3);
disp(m);                       % 3.0000 min
h = hours(2);
disp(h);                       % 2.0000 hr
dy = days(7);
disp(dy);                      % 7.0000 days

% Arithmetic: datetime + duration -> datetime.
later = t + hours(3);
disp(later);                   % 01-May-2026 15:30:00

earlier = t - minutes(15);
disp(earlier);                 % 01-May-2026 12:15:00

% datetime - datetime -> duration.
elapsed = later - t;
disp(elapsed);                 % 3.0000 hr

% duration + duration -> duration.
total = hours(2) + minutes(30);
disp(total);                   % 2.5000 hr

% duration - duration -> duration.
diff = days(7) - hours(12);
disp(diff);                    % 6.5000 days
