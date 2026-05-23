% scattering_svm.m — Wavelet Toolbox Tier-6 cross-toolbox headline.
% ----------------------------------------------------------------------
% Signal classification with wavelet scattering features + an SVM (the UG
% "Signal Classification Using Wavelet-Based Features and SVM").  Scattering
% features are time-averaged |CWT| coefficients — translation-invariant and
% discriminative; fitcsvm (Statistics Toolbox) trains the classifier.
fs = 1000;
t  = (0:255)/fs;

% two classes: low-frequency tone vs high-frequency tone (+ deterministic jitter)
f1 = waveletScattering(sin(2*pi*30*t))';
f2 = waveletScattering(sin(2*pi*30*t + 0.3) + 0.2*cos(2*pi*33*t))';
f3 = waveletScattering(sin(2*pi*150*t))';
f4 = waveletScattering(sin(2*pi*150*t + 0.3) + 0.2*cos(2*pi*160*t))';

X = [f1; f2; f3; f4];
y = [1; 1; 2; 2];
fprintf('feature matrix = %.0f x %.0f\n', size(X,1), size(X,2));

mdl = fitcsvm(X, y);
pred = predict(mdl, X);
acc = sum(pred == y) / length(y);
fprintf('training accuracy = %.2f\n', acc);
