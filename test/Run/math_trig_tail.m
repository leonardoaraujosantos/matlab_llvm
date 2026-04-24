% Trig tail + sign: asin/acos/atan/atan2/sinh/cosh/tanh/log2/log10/sign.
% Each has a scalar f64 variant and an elementwise matrix variant.

% Scalar variants.
disp(asin(1));               % 1.5708 (pi/2)
disp(acos(0));               % 1.5708
disp(atan(1));               % 0.7854 (pi/4)
disp(atan2(1, 1));           % 0.7854

disp(sinh(0));               % 0
disp(cosh(0));               % 1
disp(tanh(0));               % 0

disp(log2(8));               % 3
disp(log10(1000));           % 3

disp(sign(-7));              % -1
disp(sign(0));               % 0
disp(sign(4.2));             % 1

% Matrix variants.
A = [0 1; -1 0];
disp(sign(A));
disp(tanh(A));
