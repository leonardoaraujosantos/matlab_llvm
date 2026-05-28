% dl_gradcam.m — Deep Learning T6.1: gradient-based class attribution
% (Grad-CAM style) over the dlarray autodiff.  Train a small MLP
% classifier on a 2-D 3-class toy set, then for each test sample show
% which input dimension the picked-class score is most sensitive to.
% This is the technique under every gradient-attribution method --
% Grad-CAM, Integrated Gradients, plain saliency maps -- and ships
% essentially for free once the autodiff is in place.

rng(0);

% A 2-D / 3-class split where each class lives in its own "direction"
% in the input plane, so the saliency should clearly point at the
% deciding input dimension for each class.
Xd = [ 2.0  2.5  2.2   0.1 -0.1  0.2  -1.5 -1.8 -1.4;
       0.1  0.0 -0.1   2.0  2.4  2.2  -1.5 -1.6 -1.4 ];
Td = [ 1 1 1 0 0 0 0 0 0;
       0 0 0 1 1 1 0 0 0;
       0 0 0 0 0 0 1 1 1 ];

W1 = dlarray(0.5 * randn(8, 2)); b1 = dlarray(zeros(8, 1));
W2 = dlarray(0.5 * randn(3, 8)); b2 = dlarray(zeros(3, 1));

X = dlarray(Xd); T = dlarray(Td);

% Quick training so the model actually has class-discriminative gradients.
lr = 0.5;
for it = 1:300
    H = relu(W1 * X + b1);
    Y = softmax(W2 * H + b2);
    L = crossentropy(Y, T);
    gW1 = dlgradient(L, W1); gb1 = dlgradient(L, b1);
    gW2 = dlgradient(L, W2); gb2 = dlgradient(L, b2);
    W1 = dlarray(extractdata(W1) - lr * gW1); b1 = dlarray(extractdata(b1) - lr * gb1);
    W2 = dlarray(extractdata(W2) - lr * gW2); b2 = dlarray(extractdata(b2) - lr * gb2);
end

% Compute Grad-CAM-style saliency for one sample of each class.
% Saliency for sample i: ∇_x logit(picked_class)(x_i).
samples = [1 4 7];   % one canonical sample per class
classes = [1 2 3];

for s = 1:3
    n   = samples(s);
    cls = classes(s);
    x_i = dlarray(Xd(:, n));
    h_i = relu(W1 * x_i + b1);
    logit_i = W2 * h_i + b2;
    % Build a per-class one-hot mask + sum-of-products = picked-class logit.
    one_hot_d = zeros(3, 1);
    one_hot_d(cls) = 1.0;
    sc = sum(logit_i .* dlarray(one_hot_d));
    sal = dlgradient(sc, x_i);

    biggest = 1;
    if abs(sal(2)) > abs(sal(1)); biggest = 2; end
    fprintf('class %.0f sample dim with biggest |saliency| = %.0f\n', cls, biggest);
end
