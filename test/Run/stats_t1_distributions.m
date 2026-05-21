% Statistics Toolbox Tier-1 — probability distributions (pdf/cdf/inv).
fprintf('normpdf0   %.4f\n', normpdf(0));
fprintf('normcdf196 %.4f\n', normcdf(1.96));
fprintf('norminv975 %.4f\n', norminv(0.975));
fprintf('normcdf3   %.4f\n', normcdf(2, 0, 1));
fprintf('normpdf5   %.4f\n', normpdf(5, 5, 2));
fprintf('exppdf     %.4f\n', exppdf(1, 2));
fprintf('expcdf     %.4f\n', expcdf(1, 1));
fprintf('expinv     %.4f\n', expinv(0.5, 1));
fprintf('unifcdf    %.4f\n', unifcdf(0.5, 0, 1));
fprintf('unifinv    %.4f\n', unifinv(0.25, 0, 4));
