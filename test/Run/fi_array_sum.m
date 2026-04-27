% Phase-3 fi: sum over an fi array. The Q15 saturation behavior is part
% of the contract — sum([0.5, 0.5, 0.5, 0.5]) saturates to ~1.99994 in
% Q1.15 because 4*0.5 = 2.0 exceeds the signed Q1.15 max of 1.99994.
x = fi(0.5, 1, 16, 14);
arr = [x, x, x, x];
s = fi(0, 1, 16, 14);
s(:) = sum(arr);
disp(s);
