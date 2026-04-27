% Phase-5 fi: reinterpretcast bit-reinterprets the stored integer as a
% different numerictype. Storage width must match (both i16 here);
% only FL/signedness change.
T_in  = numerictype(1, 16, 8);
T_out = numerictype(1, 16, 2);
a = fi(1.5, T_in);          % stored = 384, real-world 1.5
b = reinterpretcast(a, T_out);
disp(int(a));   % 384
disp(int(b));   % 384 — bit-identical
disp(b);        % 384 / 2^2 = 96
