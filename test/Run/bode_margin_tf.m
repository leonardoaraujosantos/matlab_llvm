% bode 3-output [mag, phase, wout] + margin / dcgain / bandwidth on a tf model
% (the bode_first_order.m chain). wout echoes the supplied frequency grid.
G = tf(1, [0.5 1]);          % 1/(0.5 s + 1), corner at w = 2 rad/s
w = logspace(-1, 2, 200);
[mag, phase, wout] = bode(G, w);
[~, idx] = min(abs(wout - 2));
fprintf('n %.0f magc %.4f\n', numel(wout), mag(idx));   % ~0.7071 at corner
fprintf('dc %.4f bw %.3f\n', dcgain(G), bandwidth(G));  % 1.0, ~2.0

L = tf(4, [1 2 0]);          % 4 / (s^2 + 2s) — open-loop type-1
[Gm, Pm, Wcg, Wcp] = margin(L);
fprintf('Pm %.2f Wcp %.3f\n', Pm, Wcp);                 % ~51.8 deg, ~1.57
