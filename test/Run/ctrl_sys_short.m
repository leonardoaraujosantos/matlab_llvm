% Model-object short forms — §3.1 follow-on.
%
% A class-pinned first argument routes to the matching matrix-arg
% primitive by unpacking the relevant properties via
% matlab_obj_get_mat:
%
%   pole(sys)        → eig(sys.A)             (ss)
%                    → roots(sys.Denominator) (tf)
%   dcgain(sys)      → dcgain_ss(A, B, C, D)  (ss only)
%   bandwidth(sys)   → bandwidth_ss(...)      (ss only)
%   step(sys)        → step_ss(A, B, C, D, dt=0.01, N=500) (ss)
%   step(sys, dt, N) → step_ss with explicit dt / N
%   lsim(sys, u, dt) → lsim_ss(A, B, C, D, u, dt)
%   bode(sys, w)     → bode_ss(A, B, C, D, w) (ss, mag only)
%                    → bode_tf(num, den, w)   (tf, mag only)
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_model_objects.

A = [-1 0; 0 -2];
B = [1; 1];
C = [1 0];
D = 0;
sys = ss(A, B, C, D);

% --- pole(ss): eigenvalues of A.
disp(pole(sys));

% --- dcgain(ss).
disp(dcgain(sys));

% --- bandwidth(ss): -3 dB frequency.
disp(bandwidth(sys));

% --- step(ss) with default dt/N → 500-sample column.
y = step(sys);
disp(size(y));

% --- bode(ss, w): magnitude at sample frequencies.
w = [0.1; 1; 10];
mag = bode(sys, w);
disp(mag);

% --- tf side.
G = tf([1 2], [1 3 5]);
disp(pole(G));
mag_tf = bode(G, w);
disp(mag_tf);
