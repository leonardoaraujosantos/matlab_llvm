% Typed-driver-call path, unsigned 8-bit. The function takes args
% with no pragma; the typed driver passes `fi(_, 0, 8, 0)` values
% and the user-call refinement should propagate the unsignedness
% to the func args. Without the threading, `en` and `rst` would
% emit as `logic signed [7:0]` (the pre-fix bug).
T = numerictype(0, 8, 0);
y = driver_unsigned8(fi(1, T), fi(0, T));
disp(y);

function count = driver_unsigned8(en, rst)
    persistent c;
    if isempty(c)
        c = fi(0, numerictype(0, 8, 0));
    end
    z = fi(0, numerictype(0, 8, 0));
    if rst > z
        c = fi(0, numerictype(0, 8, 0));
    elseif en > z
        c = c + fi(1, numerictype(0, 8, 0));
    end
    count = c;
end
