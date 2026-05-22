% step honouring a supplied time vector + the [y, tout] 2-output, on both
% tf and ss models. Also exercises scalar `^` on the AOT path (wn^2 in the
% tf coefficients), which previously left an unconverted matlab.matpow.
G = tf(1, [0.5 1]);          % 1/(0.5 s + 1), tau = 0.5
t = 0:0.01:3;
[y, tout] = step(G, t);
fprintf('n %.0f\n', numel(tout));     % 301 (tout echoes t)
fprintf('ytau %.4f\n', y(51));        % 1 - 1/e = 0.6321 at t = tau
fprintf('tout %.2f\n', tout(51));     % 0.50

wn = 1.0; zeta = 0.5;
G2 = tf(wn^2, [1, 2*zeta*wn, wn^2]);  % scalar `^` in the coefficients
t2 = 0:0.05:15;
y2 = step(G2, t2);
fprintf('peak %.3f\n', max(y2));      % 1.163
fprintf('final %.3f\n', y2(end));     % 0.999

A = [0 1; 0 0]; B = [0; 1]; C = [1 0]; D = 0;
sys = ss(A, B, C, D);
t3 = 0:0.1:5;
y3 = step(sys, t3);                   % step on ss, honouring t3
fprintf('di %.1f\n', y3(21));         % double integrator: y(t=2) = 2.0
