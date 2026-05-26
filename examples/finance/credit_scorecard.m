% Financial Toolbox Tier-4 §3 — credit scorecard (logistic core).
% Two predictors that separate defaulters: low score1 + high score2
% => higher default probability.

% Training data: 10 obs, 2 predictors, binary default flag.
X = [ 7.0 1.0
      6.5 1.5
      8.0 0.5
      3.0 4.0
      2.5 4.5
      3.5 3.8
      7.5 0.8
      2.0 5.0
      6.8 1.2
      3.2 4.2 ];
% Defaults concentrated in the low-score1/high-score2 cluster.
y = [0; 0; 0; 1; 1; 1; 0; 1; 0; 1];

sc = creditscorecard(X, y);
sc = fitmodel(sc);

% Predict on the training set; PD should be high for the defaulter
% rows (4,5,6,8,10) and low for the rest.
pd = probdefault(sc, X);
fprintf('PD(good row 1) = %.3f\n', pd(1));   % near 0
fprintf('PD(bad  row 5) = %.3f\n', pd(5));   % near 1

% Score (log-odds) ordering: bad rows score higher.
s = score(sc, X);
fprintf('score gap row5-row1 = %.2f\n', s(5) - s(1));   % positive

% A fresh borrower clearly in the "good" region.
xnew = [7.2 0.9];
pdn = probdefault(sc, xnew);
fprintf('PD(new good applicant) = %.3f\n', pdn(1));
