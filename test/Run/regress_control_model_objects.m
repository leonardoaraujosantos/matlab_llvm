% regress_control_model_objects.m — #293: tf/ss model objects support
% tfdata / ssdata extraction, c2d discretization, and disp pretty-printing.
% Locks the behavior (filed when examples/control/tf_basic.m was marked
% "NOT YET SHIPPED"; the capability has since shipped).

% --- tfdata: numerator / denominator coefficient rows --------------
G = tf([1 2], [1 3 5]);
[num, den] = tfdata(G);
disp(num);
disp(den);

% --- ssdata: A/B/C/D matrices --------------------------------------
sys = ss([0 1; -2 -3], [0; 1], [1 0], 0);
[A, B, C, D] = ssdata(sys);
disp(A);
disp(B);
disp(C);

% --- c2d: discretize a tf (ZOH) ------------------------------------
Gd = c2d(tf(1, [1 1]), 0.1);
[~, dend] = tfdata(Gd);
disp(dend);

% --- disp(tf): s-domain rendering ----------------------------------
disp(G);
