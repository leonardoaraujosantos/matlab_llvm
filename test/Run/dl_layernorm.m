% dl_layernorm.m — LayerNorm headline.  Composes the Phase-1 small ops:
%   mu   = mean(X, 1)              % feature-axis mean       (1xN)
%   diff = X - mu                  % broadcasted              (MxN)
%   var  = mean(diff .* diff, 1)   % feature-axis variance    (1xN)
%   y    = diff ./ sqrt(var + eps) * gamma + beta
% Pinned `eps_dl` is a dlarray with constant 1e-5 so `v + eps_dl` flows
% through the dlarray-plus dispatch (mixing scalars with dlarrays in the
% same expression keeps Sema on the numeric lane).

% Toy batch: 3 features × 4 examples.
X      = dlarray([1.0  2.0  0.5  3.0;
                  0.5  1.5  1.0  2.5;
                  2.0  3.0  2.5  4.0]);
gamma  = dlarray([1.0  1.0  1.0  1.0]);
beta   = dlarray([0.0  0.0  0.0  0.0]);
eps_dl = dlarray([1e-5 1e-5 1e-5 1e-5]);

% --- forward -------------------------------------------------------------
mu     = mean(X, 1);                       % 1x4 row mean
diff   = X - mu;                           % 3x4 (matrix - row broadcast)
sqdiff = diff .* diff;                     % 3x4
v      = mean(sqdiff, 1);                  % 1x4
denm   = sqrt(v + eps_dl);                 % 1x4
xhat   = diff ./ denm;                     % 3x4 (row broadcast)
scaled = xhat .* gamma;                    % 3x4 (row broadcast)
Y      = scaled + beta;                    % 3x4 (row broadcast)

% --- read out ------------------------------------------------------------
Yv = extractdata(Y);
% Per-column mean / variance from the raw numeric matrix.
m_per_col = sum(Yv, 1) / 3;
v_per_col = sum(Yv .* Yv, 1) / 3 - m_per_col .* m_per_col;
fprintf('dl_layernorm: sum(|per-col mean|) = %.4e\n', sum(abs(m_per_col)));
fprintf('dl_layernorm: mean(per-col var)   = %.4f\n', sum(v_per_col) / 4);

% --- gradient sanity -----------------------------------------------------
L  = sum(sum(Y));
gG = dlgradient(L, gamma);
gB = dlgradient(L, beta);
fprintf('dl_layernorm: sum(gG)=%.4f sum(gB)=%.4f\n', sum(gG), sum(gB));
fprintf('dl_layernorm: PASS\n');
