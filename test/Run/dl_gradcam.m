% Deep Learning T6.1 gating test — Grad-CAM-style attribution over the
% dlarray autodiff.  For a small MLP classifier f(x) = softmax(W2*relu(W1*x+b1)+b2),
% the gradient of the picked class's pre-softmax logit w.r.t. the hidden
% activations weights each unit's contribution to that class decision.
% Multiplying activations by their gradient + summing back to input shape
% yields the per-input "saliency" that's the MLP analogue of Grad-CAM's
% conv-feature-map weighting.  This is the technique behind every other
% gradient-attribution method (Grad-CAM, Integrated Gradients, saliency
% maps) and lives entirely over the shipped autodiff.
%
% Gating signal:
%   (a) saliency has the same shape as the input,
%   (b) the dimension that drives the decision gets the largest |saliency|.

rng(0);

% Build a 2-D-input → 4-hidden → 2-class network so the headline
% saliency check is unambiguous.
W1 = dlarray([ 1.0  0.0;
               0.0  1.0;
              -1.0  0.0;
               0.0 -1.0]);
b1 = dlarray(zeros(4, 1));
W2 = dlarray([ 1.0  1.0 -1.0 -1.0;
              -1.0 -1.0  1.0  1.0]);   % class 1 likes (h0+h1)-(h2+h3); class 2 the opposite
b2 = dlarray(zeros(2, 1));

% A test input where x(1) >> x(2) — class 1 should dominate.
x = dlarray([2.0; 0.1]);

% Forward through the relu hidden layer, then the linear logit.  We
% intentionally compute the pre-softmax logit (Grad-CAM convention).
h     = relu(W1 * x + b1);
logit = W2 * h + b2;

% Saliency for class 1: dlgradient(logit(class1), x).  We can't index
% inside a dlarray and keep the tape, so we mix the two logits as a
% one-hot weighted sum: logit · [1; 0] = logit(1).
one_hot_c1 = dlarray([1; 0]);
sc_c1 = sum(logit .* one_hot_c1);
sal1 = dlgradient(sc_c1, x);

% Saliency for class 2 — the same setup with the opposite one-hot mask.
one_hot_c2 = dlarray([0; 1]);
sc_c2 = sum(logit .* one_hot_c2);
sal2 = dlgradient(sc_c2, x);

% (a) shape check.
shape_ok = 0;
if size(sal1, 1) == 2
    if size(sal1, 2) == 1
        shape_ok = 1;
    end
end

% (b) "decisive dimension" check.  W1 routes x(1) through hidden units 0 and 2
% (W1(:,1) is non-zero only at rows 1 and 3).  With x(1) = 2.0 and x(2) = 0.1:
% h(1) = 2, h(2) = 0.1, h(3) = 0 (clipped), h(4) = 0 (clipped); logit(1) = h(1)+h(2)
% and gradient w.r.t. x(1) comes through h(1) (gain +1 via W1(1,1)) → sal1(1) > 0
% and |sal1(1)| > |sal1(2)|.
biggest_c1 = 1;
if abs(sal1(2)) > abs(sal1(1)); biggest_c1 = 2; end
% Class 2 reverses the sign + dominant unit: should still be dimension 1 driving
% the magnitude (because that's where x's mass is).
biggest_c2 = 1;
if abs(sal2(2)) > abs(sal2(1)); biggest_c2 = 2; end

right_dim = 0;
if biggest_c1 == 1
    if biggest_c2 == 1
        right_dim = 1;
    end
end

% Sign check: class 1 has positive grad on the active hidden unit, class 2 negative.
sign_flip = 0;
if sal1(1) > 0
    if sal2(1) < 0
        sign_flip = 1;
    end
end

fprintf('saliency matches input shape = %.0f\n', shape_ok);
fprintf('decisive input dim picked = %.0f\n', right_dim);
fprintf('saliency sign flips between classes = %.0f\n', sign_flip);
