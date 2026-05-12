% timeresp on a known rational: step response of H(s) = 1/(s + 1).
%
%   y(t) = 1 - e^{-t}.
%
% Fit (just feed the analytic poles/residues directly via rationalfit
% on synthetic samples), then step through with ts = 0.1 s.

K = 20;
freqs = zeros(K, 1);
for k = 1:K
    freqs(k) = 0.01 + (k - 1) * 0.05;       % low-freq grid (Hz)
end

% H(s) = 1/(s + 1), evaluated at s = j·2π·f:
h_re = zeros(K, 1);
h_im = zeros(K, 1);
for k = 1:K
    w = 2.0 * 3.141592653589793 * freqs(k);
    den = 1.0 + w*w;
    h_re(k) = 1.0 / den;
    h_im(k) = -w / den;
end

mdl = rationalfit(freqs, h_re, h_im, 1, 10);

% Drive with a unit step input.  Step response: y(t) = 1 - e^{-t}.
N = 5;
u = ones(N, 1);
y = timeresp(mdl, u, 1.0);    % ts = 1 second.
disp(y);
% Expected: y[0]=0, y[1]=1-1/e ≈ 0.632, y[2]=1-1/e² ≈ 0.865,
%           y[3]=1-1/e³ ≈ 0.950, y[4]=1-1/e⁴ ≈ 0.982.
