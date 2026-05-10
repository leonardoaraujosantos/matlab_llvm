% disp(tf) — formatted s-domain rendering.
%
% The Lowering.cpp `disp(obj)` class-method dispatch routes a
% tf-pinned operand through `matlab_tf_disp` in the runtime. The
% helper pulls Numerator / Denominator off the matlab_obj, formats
% each as a polynomial string ("s + 2", "s^2 + 3 s + 5"), and prints
% the centered fraction with a "Continuous-time transfer function."
% trailer — the same shape MATLAB's `display(tf)` uses.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_tf_basic
% (class struct passed by value vs. void* runtime ABI).

G = tf([1 2], [1 3 5]);
disp(G);

H = tf([1], [1 1]);
disp(H);

J = tf([2 0 1], [1 0 0 4]);
disp(J);

% disp(tf) on a tf composed from operator overloads.
s = tf('s');
K = (s + 2) / (s * s + 3 * s + 5);
disp(K);
