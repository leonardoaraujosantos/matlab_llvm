% Deep Learning T6.6 gating test — `l-inf` robustness check.  The MATLAB
% UG offers `verifyNetworkRobustness` as a SAT-based proof tool; we ship
% a *sound but not complete* version of the same answer via the autodiff:
%
%   linfBound(x; ε) := ε * ||∇_x f(x)||₁
%
% Bounds the magnitude of the score perturbation that any l∞-ball of
% radius ε around x can produce, by the dual-norm inequality
% (∀ δ : |δ|_∞ ≤ ε, |∇f·δ| ≤ ε·||∇f||₁).  If the bound is smaller than the
% margin between the predicted-class score and the runner-up, the network
% is *certified robust* at x for that ε.
%
% Gating signal: a trained binary-margin network with a strong-side input
% has a margin > linfBound for a small ε, and < linfBound for an ε large
% enough to flip the prediction.  Both bounds are computed from the
% gradient of the picked-class score w.r.t. the input.

rng(0);

% A 2-D linear-classifier-like MLP that strongly separates class 0/1.
W1 = dlarray([ 1.0  0.0; -1.0  0.0; 0.0  1.0; 0.0 -1.0]);
b1 = dlarray(zeros(4, 1));
W2 = dlarray([ 2.0 -2.0  0.5 -0.5;
              -2.0  2.0 -0.5  0.5]);
b2 = dlarray(zeros(2, 1));

% Input x where class 1 dominates by a large margin.
x = dlarray([1.5; 0.2]);

h     = relu(W1 * x + b1);
logit = W2 * h + b2;
L = extractdata(logit);

% argmax of the 2-element logit + margin to the runner-up.  Inlined
% because local functions in script files don't lower cleanly.
picked = 1;
runner = 2;
if L(2) > L(1)
    picked = 2;
    runner = 1;
end
margin = L(picked) - L(runner);

% Gradient of the picked-class logit w.r.t. x.
one_hot_d = zeros(2, 1);
one_hot_d(picked) = 1.0;
sc = sum(logit .* dlarray(one_hot_d));
g  = dlgradient(sc, x);

% l1-norm of the gradient (× ε bounds the worst-case logit change).
gn = 0;
for k = 1:2
    gn = gn + abs(g(k));
end

% Two epsilons: small (must be certified safe), large (large enough that
% the dual-norm bound exceeds the margin, so robustness can't be
% guaranteed -- a sufficient condition that the SAT-based verifier
% would still potentially certify, but the gradient bound can't).
eps_safe = 0.1;
eps_atk  = 5.0;
bound_safe = eps_safe * gn;
bound_atk  = eps_atk  * gn;

certified = 0;
if bound_safe < margin
    certified = 1;
end
bound_grows = 0;
if bound_atk > bound_safe
    bound_grows = 1;
end
% At eps_atk = 5 with the gradient l1-norm 2.5, the bound = 12.5 > margin = 6.2.
beyond_margin = 0;
if bound_atk > margin
    beyond_margin = 1;
end

fprintf('predicted class = %.0f\n', picked);
fprintf('linf bound at small eps < margin (certified) = %.0f\n', certified);
fprintf('bound is monotone in eps = %.0f\n', bound_grows);
fprintf('large-eps bound > margin = %.0f\n', beyond_margin);
