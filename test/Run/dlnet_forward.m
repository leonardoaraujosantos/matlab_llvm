% Deep Learning Tier-1 — dlarray forward inference (2-layer MLP).
W1 = dlarray([0.1 0.2; 0.3 0.4; -0.1 0.5]);   % 3x2
b1 = dlarray([0.0; 0.1; -0.1]);
W2 = dlarray([0.2 -0.3 0.1; 0.4 0.5 -0.2]);    % 2x3
b2 = dlarray([0.0; 0.0]);
x  = dlarray([1.0; 2.0]);
hd = extractdata(relu(W1*x + b1));
fprintf('hidden relu = %.4f %.4f %.4f\n', hd(1), hd(2), hd(3));
y  = softmax(W2*relu(W1*x + b1) + b2);
yd = extractdata(y);
fprintf('softmax out = %.4f %.4f (sum %.4f)\n', yd(1), yd(2), yd(1)+yd(2));
