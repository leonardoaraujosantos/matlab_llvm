% Navigation Tier-5 — stateEstimatorPF generic linear-Gaussian PF.
rng(0);
pf = stateEstimatorPF();
initialize(pf, 2000, [0 0], [4 4]);
A = [1 0; 0 1];
for k = 1:8
    predict(pf, A, [0.05 0.05]);
    correct(pf, [5 3], [1 0; 0 1], [0.4 0.4]);
end
est = getStateEstimate(pf);
fprintf('PF estimate ~ (5,3): (%.0f,%.0f)\n', round(est(1)), round(est(2)));
