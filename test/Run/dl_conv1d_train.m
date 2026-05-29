% dl_conv1d_train — 1-D convolution training over the dlarray autodiff
% tape, unlocking convolution1dLayer / sequenceInputLayer carve-down
% items.
%
% 1-D conv is a height-1 2-D conv: a (T, C_in, N) sequence input is
% reshaped to (1, T, C_in, N) and passed through the shipped
% conv2d_batch (which has OP_CONV2D_BATCH backward).  Kernel is
% (1, kT, C_in, C_out).
%
% Task: train a single 1-D conv (kT=3, C_in=2, C_out=1) to detect a
% 3-element rising-ramp pattern in a 6-element sequence.  Verify the
% loss drops sharply — gradient flows through the rank-4 conv backward
% correctly when the height dim is 1.

T = 6;
C_in = 2;
C_out = 1;
kT = 3;
N = 2;        % batch
Tout = T - kT + 1;  % 4

% Inputs: two sequences, two channels each.
% Sample 1: channel 1 is a rising ramp, channel 2 is flat.
% Sample 2: channel 1 is flat, channel 2 is a rising ramp.
% Target: detect rising-ramps in channel 1 -> high response on sample 1.
X = zeros(1, T, C_in, N);
X(1, 1, 1, 1) = 0.0; X(1, 2, 1, 1) = 0.3; X(1, 3, 1, 1) = 0.6;
X(1, 4, 1, 1) = 0.9; X(1, 5, 1, 1) = 1.2; X(1, 6, 1, 1) = 1.5;
% Channel 2 of sample 1 — flat 0.5.
for t = 1:T, X(1, t, 2, 1) = 0.5; end
% Sample 2 — opposite.
for t = 1:T, X(1, t, 1, 2) = 0.5; end
X(1, 1, 2, 2) = 0.0; X(1, 2, 2, 2) = 0.3; X(1, 3, 2, 2) = 0.6;
X(1, 4, 2, 2) = 0.9; X(1, 5, 2, 2) = 1.2; X(1, 6, 2, 2) = 1.5;

% Target: single output channel, high on sample 1 (channel-1-ramp
% detector), low on sample 2.
T_target = zeros(1, Tout, C_out, N);
for t = 1:Tout
    T_target(1, t, 1, 1) = 1.0;
    T_target(1, t, 1, 2) = 0.0;
end

% Initial 1-D conv kernel: (1, kT, C_in, C_out) — fill with a
% deterministic small-but-non-zero pattern so training has signal but
% starts far from the optimum.
W = zeros(1, kT, C_in, C_out);
W(1, 1, 1, 1) = 0.10; W(1, 2, 1, 1) = -0.05; W(1, 3, 1, 1) = 0.02;
W(1, 1, 2, 1) = -0.03; W(1, 2, 2, 1) = 0.07; W(1, 3, 2, 1) = -0.04;

lr = 0.05;
n_iter = 200;
Xdl = dlarray(X);
Tdl = dlarray(T_target);
Wdl = dlarray(W);
L0 = extractdata(dlarray(0.0));
L_last = extractdata(dlarray(0.0));
for it = 1:n_iter
    Y = conv2d_batch(Xdl, Wdl);
    loss = mse(Y, Tdl);

    Lv = extractdata(loss);
    if it == 1
        L0 = Lv;
    end
    L_last = Lv;

    gW = dlgradient(loss, Wdl);
    Wdl = dlarray(extractdata(Wdl) - lr * gW);
end

fprintf('dl_conv1d_train: loss(0) = %.4f\n', L0);
fprintf('dl_conv1d_train: loss(%d) = %.6f\n', n_iter, L_last);
fprintf('dl_conv1d_train: ratio = %.6f\n', L_last / L0);

if L_last < L0 * 0.1
    fprintf('dl_conv1d_train: PASS\n');
else
    fprintf('dl_conv1d_train: FAIL\n');
end
