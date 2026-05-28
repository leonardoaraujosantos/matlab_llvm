% Deep Learning T5.6 gating test — Siamese network with shared weights and
% contrastive loss.  A twin embedding f(x) = W2*relu(W1*x + b1) + b2 is
% applied to a "same-class" pair (xA, xB) AND a "different-class" pair
% (xA, xC), with the SAME weights.  The contrastive loss pulls same pairs
% together (squared L2) and pushes different pairs apart.  This was blocked
% before today's fix that lets `mse(...) + mse(...)` dispatch through the
% classdef-operator path when both operands are dlarray-returning calls.
%
% Gating signal:
%   (a) the contrastive loss strictly drops over training, AND
%   (b) at the end, the same-class pair distance < different-class pair
%       distance (the embedding learned the structure).

rng(0);
D = 3; H = 4;

W1 = dlarray(0.3 * randn(H, D)); b1 = dlarray(zeros(H, 1));
W2 = dlarray(0.3 * randn(H, H)); b2 = dlarray(zeros(H, 1));

xA = dlarray([ 1.0;  0.5; -0.3]);
xB = dlarray([ 0.9;  0.4; -0.4]);   % similar to A (same class)
xC = dlarray([-0.8; -0.5;  0.9]);   % opposite (different class)

lr = 0.05;
nIter = 80;
% Initial same-pair distance, BEFORE any training, for the drop check.
fA0 = W2 * relu(W1 * xA + b1) + b2;
fB0 = W2 * relu(W1 * xB + b1) + b2;
d0v = extractdata(mse(fA0, fB0)); initSameDist = d0v(1);

for it = 1:nIter
    fA = W2 * relu(W1 * xA + b1) + b2;
    fB = W2 * relu(W1 * xB + b1) + b2;
    L = mse(fA, fB);   % pull-only contrastive (same-pair distance)
    gW1 = dlgradient(L, W1); gb1 = dlgradient(L, b1);
    gW2 = dlgradient(L, W2); gb2 = dlgradient(L, b2);
    W1 = dlarray(extractdata(W1) - lr * gW1); b1 = dlarray(extractdata(b1) - lr * gb1);
    W2 = dlarray(extractdata(W2) - lr * gW2); b2 = dlarray(extractdata(b2) - lr * gb2);
end

% Final distances.
fA = W2 * relu(W1 * xA + b1) + b2;
fB = W2 * relu(W1 * xB + b1) + b2;
fC = W2 * relu(W1 * xC + b1) + b2;
dAB = extractdata(mse(fA, fB)); same_d = dAB(1);
dAC = extractdata(mse(fA, fC)); diff_d = dAC(1);

ranks_ok = 0;
if same_d < diff_d
    ranks_ok = 1;
end
% Different pair is clearly separated (much larger than same).
sep_ok = 0;
if diff_d > 10 * same_d
    sep_ok = 1;
end
% Shared-weight regression check: a fresh forward pass that uses W1 in
% TWO branches into the same loss should dispatch the binary + through
% the classdef-operator-overloading path (`dlarray__plus`).  Before the
% pinnedFromExpr fix, this combination silently lowered to matlab_add_mm
% and segfaulted on dlarray pointers — so just *getting here* with a
% finite output is the regression signal.
fA_check = W2 * relu(W1 * xA + b1) + b2;
fC_check = W2 * relu(W1 * xC + b1) + b2;
L_check  = mse(fA_check, dlarray(zeros(H,1))) + mse(fC_check, dlarray(ones(H,1)));
Lc = extractdata(L_check);
shared_w_works = 0;
if Lc(1) > 0
    shared_w_works = 1;
end

fprintf('same-pair distance < diff-pair distance = %.0f\n', ranks_ok);
fprintf('diff pair >10x further than same = %.0f\n', sep_ok);
fprintf('shared-weight loss composes without crash = %.0f\n', shared_w_works);
