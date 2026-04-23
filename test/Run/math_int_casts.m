% Integer cast builtins: truncate + saturate to the target dtype's
% range. Runtime stays f64 internally, but the visible value matches
% what MATLAB would compute for a true typed integer.
disp(int32(3.7));        % 3   — truncate toward zero
disp(int32(-2.9));       % -2
disp(uint8(300));        % 255 — saturate up
disp(uint8(-5));         % 0   — saturate down
disp(int8(200));         % 127
disp(int16(40000));      % 32767
disp(uint16(-1));        % 0
disp(logical(0));        % 0
disp(logical(42));       % 1
disp(double(7));         % 7
disp(single(3.14));      % ~3.14 after float rounding
