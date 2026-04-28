% Typed-driver-call path, signed 16-bit. Mirror of
% driver_unsigned8 with signed=1 in the numerictype, asserting
% the call-refinement attr-threading also produces `signed` when
% the source spec is signed.
T = numerictype(1, 16, 0);
y = driver_signed16(fi(5, T), fi(3, T));
disp(y);

function r = driver_signed16(a, b)
    r = a + b;
end
