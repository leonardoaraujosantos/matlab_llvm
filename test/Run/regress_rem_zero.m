% Regression: rem(x,0) is NaN in MATLAB (mod(x,0) is x). Previously rem(x,0)
% returned x. NaN is detected backend-independently via r==r being false.
r = rem(5, 0);
if r == r; disp(1); else; disp(0); end     % NaN -> 0
rn = rem(-5, 0);
if rn == rn; disp(1); else; disp(0); end    % NaN -> 0
rm = rem([4 6], 0);                          % matrix form: each element NaN
e1 = rm(1);
if e1 == e1; disp(1); else; disp(0); end     % 0
e2 = rm(2);
if e2 == e2; disp(1); else; disp(0); end     % 0
fprintf('%.0f %.0f\n', rem(7,3), rem(-7,3)); % nonzero divisor unchanged: 1 -1
fprintf('%.0f\n', mod(5,0));                 % mod(x,0) unchanged: 5
