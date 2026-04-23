% No error: catch body skipped.
x = 0;
try
    x = 1;
catch
    x = 99;
end
disp(x);

% Explicit error(): catch body runs.
y = 0;
try
    error('boom');
    y = 1;
catch
    y = 99;
end
disp(y);

% Nested try/catch isn't tested here — v1 only handles top-level flags.
