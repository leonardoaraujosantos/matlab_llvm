% Deep Learning Toolbox Tier-2 — reverse-mode autodiff vs finite difference.
% loss = sum(sigmoid(W*x)); dlgradient sweeps the tape for dL/dW.
% (Perturbations use matrix addition, which allocates a fresh matrix — the
% `B = A; B(i)=...` copy-on-write path has a separate known limitation.)
Wd = [0.5 -0.3; 0.2 0.8];
xd = [1.0; 2.0];
W  = dlarray(Wd);
x  = dlarray(xd);
loss = sum(sigmoid(W*x));
g = dlgradient(loss, W);
fprintf('analytic dL/dW = [%.5f %.5f; %.5f %.5f]\n', g(1,1), g(1,2), g(2,1), g(2,2));

ep = 1e-6;
maxerr = 0;
for i = 1:2
    for j = 1:2
        E = zeros(2, 2);
        E(i, j) = ep;
        lp = extractdata(sum(sigmoid(dlarray(Wd + E) * x)));
        lm = extractdata(sum(sigmoid(dlarray(Wd - E) * x)));
        fd = (lp(1) - lm(1)) / (2 * ep);
        e = abs(fd - g(i, j));
        if e > maxerr
            maxerr = e;
        end
    end
end
fprintf('max |analytic - finite-diff| = %.2e\n', maxerr);
