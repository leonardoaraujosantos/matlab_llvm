% Deep Learning Toolbox Tier-3 — train an MLP from scratch over the dlarray
rng(0);
% Toy 3-class set (2 features, 6 samples), well-separated.
Xd = [1 1.2 4 4.2 2.5 2.3;
      1 0.8 1 0.8 4 4.2];
Td = [1 1 0 0 0 0;
      0 0 1 1 0 0;
      0 0 0 0 1 1];
labels = [1 1 2 2 3 3];
X = dlarray(Xd);
T = dlarray(Td);
W1 = dlarray(0.5*randn(8,2)); b1 = dlarray(zeros(8,1));
W2 = dlarray(0.5*randn(3,8)); b2 = dlarray(zeros(3,1));
lr = 0.5;
for it = 1:300
    H = relu(W1*X + b1);
    Y = softmax(W2*H + b2);
    loss = crossentropy(Y, T);
    gW1 = dlgradient(loss, W1);
    gb1 = dlgradient(loss, b1);
    gW2 = dlgradient(loss, W2);
    gb2 = dlgradient(loss, b2);
    W1 = dlarray(extractdata(W1) - lr*gW1);
    b1 = dlarray(extractdata(b1) - lr*gb1);
    W2 = dlarray(extractdata(W2) - lr*gW2);
    b2 = dlarray(extractdata(b2) - lr*gb2);
    if it == 1
        fprintf('initial loss = %.4f\n', extractdata(loss));
    end
end
finalLoss = extractdata(crossentropy(softmax(W2*relu(W1*X+b1)+b2), T));
fprintf('final loss   = %.4f\n', finalLoss);
Yf = extractdata(softmax(W2*relu(W1*X+b1)+b2));
correct = 0;
for j = 1:6
    pred = 1; best = Yf(1,j);
    for k = 2:3
        if Yf(k,j) > best
            best = Yf(k,j); pred = k;
        end
    end
    if pred == labels(j)
        correct = correct + 1;
    end
end
fprintf('train accuracy = %.0f%% (%.0f/6)\n', 100*correct/6, correct);
