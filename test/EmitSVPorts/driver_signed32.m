% Typed-driver-call path, signed 32-bit. Asserts the storage
% class width (i32) matches the declared WL=32 — no over-padding
% to the next native int.
T = numerictype(1, 32, 0);
y = driver_signed32(fi(100, T), fi(7, T));
disp(y);

function r = driver_signed32(a, b)
    r = a + b;
end
