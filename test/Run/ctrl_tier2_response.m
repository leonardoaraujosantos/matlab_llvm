% Tier-2 follow-ons — time-domain free response + frequency-domain
% extensions. Backs the matrix-arg runtime entries (impulse_ss,
% initial_ss, freqresp_ss / freqresp_tf, nyquist_ss / nyquist_tf,
% allmargin_ss, logspace) and the model-object short-form dispatch
% in Lowering.cpp.
%
% LLVM-lane only — same emit-c/cpp/python/ts skip as ctrl_sys_short.

% Two-state diagonal plant so x0 column vectors round-trip cleanly.
A = [-2 0; 0 -2];
B = [1; 1];
C = [1 0];
D = [0];
sys = ss(A, B, C, D);

% --- impulse(sys) — default dt = 0.01, N = 500.
y_imp = impulse(sys);
disp(size(y_imp));
disp(y_imp(1));     % C·B = 1
disp(y_imp(11));    % ≈ e^{-0.2} ≈ 0.8187

% --- initial(sys, x0) — free response from non-zero initial state.
x0 = [3; 0];
y_init = initial(sys, x0);
disp(y_init(1));    % = C·x0 = 3
disp(y_init(11));   % = 3·e^{-0.2} ≈ 2.456

% --- isstable(sys) / damp(sys) — pole characterisation.
disp(isstable(sys));    % 1: both poles Hurwitz
disp(damp(sys));        % [wn, zeta] per pole = [[2, 1]; [2, 1]]

% --- Frequency-domain follow-ons. First-order G(s) = 1/(s + 1).
A2 = [-1];
B2 = [1];
C2 = [1];
D2 = [0];
G = ss(A2, B2, C2, D2);
w = [0.1; 1; 10];

% freqresp(sys, w) returns matlab_mat_c (complex column).
H = freqresp(G, w);
disp(H);

% nyquist(sys, w) — N×2 real matrix [re, im]
ri = nyquist(G, w);
disp(ri);

% logspace + allmargin(sys, w) — full-grid scan.
ww = logspace(-2, 2, 9);
mar = allmargin(G, ww);
disp(mar);
