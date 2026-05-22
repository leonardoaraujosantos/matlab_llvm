% ssdata / tfdata — multi-return function-style class-method dispatch
% (`[A,B,C,D] = ssdata(sys)` routes to the ss classdef method).
sys = ss([1 2; 3 4], [1; 1], [1 0], 0);
[A, B, C, D] = ssdata(sys);
fprintf('A %.0f %.0f\n', A(1,1), A(2,1));
fprintf('B %.0f C %.0f\n', B(1), C(1));

G = tf([1 2], [1 3 5]);
[num, den] = tfdata(G);
fprintf('num %.0f %.0f\n', num(1), num(2));
fprintf('den %.0f %.0f %.0f\n', den(1), den(2), den(3));
