% Typed-driver-call path, unsigned 16-bit. Asserts the threading
% works at WL=16 too (the canonical width for many DSP / image
% pipelines).
T = numerictype(0, 16, 0);
y = driver_unsigned16(fi(40000, T), fi(20000, T));
disp(y);

function r = driver_unsigned16(a, b)
    r = a + b;
end
