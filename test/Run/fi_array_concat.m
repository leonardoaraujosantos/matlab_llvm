% Phase-3 fi: vector concat exercises the [scalar, slice] shape used in
% the FIR delay-line shift (`[x, delay(1:end-1)]`).
a = fi(zeros(1, 4), 1, 16, 14);
x = fi(0.25, 1, 16, 14);
b = [x, a(1:3)];
disp(length(b));
disp(b(1));   % 0.25
disp(b(2));   % 0
