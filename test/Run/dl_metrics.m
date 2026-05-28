% Deep Learning T6.3 gating test — classification metrics over the
% existing confusionmat kernel: accuracy / precision / recall / fScore /
% rocmetrics / aucroc.  All take (ytrue, ypred) of integer-coded labels
% (the score-based variants take a continuous score vector + the positive
% class label).

% 8-sample three-class set with one swapped label in each off-diagonal.
yt = [1; 1; 2; 2; 1; 3; 3; 2];
yp = [1; 1; 2; 1; 1; 3; 2; 2];

a = accuracy(yt, yp);
p = precision(yt, yp);
r = recall(yt, yp);
f = fScore(yt, yp);

% Accuracy = 6/8 = 0.75.  Precision/Recall/F-score per class:
%   class 1: P = 3/4, R = 3/3, F = 6/7 ≈ 0.857
%   class 2: P = 2/3, R = 2/3, F = 0.667
%   class 3: P = 1/1, R = 1/2, F = 0.667
fprintf('accuracy x100 = %.0f\n', round(100 * a(1)));
fprintf('precision class1 x100 = %.0f\n', round(100 * p(1)));
fprintf('recall class2 x100 = %.0f\n', round(100 * r(2)));
fprintf('fscore class3 x100 = %.0f\n', round(100 * f(3)));

% Binary-classification ROC + AUC.  3 positives (label 1), 3 negatives
% (label 0); scores rank 4 of 6 perfectly, with one rank flip → AUC = 7/9.
scores = [0.9; 0.7; 0.8; 0.2; 0.4; 0.1];
yroc   = [1;   1;   0;   0;   1;   0  ];
auc    = aucroc(scores, yroc, 1);

fprintf('aucroc x100 = %.0f\n', round(100 * auc(1)));
