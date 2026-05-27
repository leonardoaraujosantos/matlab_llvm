% Econometrics Toolbox — Tier-5 headline.
% A local-level state-space model recovers a smooth latent signal from
% noisy observations using the Kalman filter + RTS smoother.
%   level_t = level_{t-1} + w_t     (random-walk state)
%   y_t     = level_t     + v_t     (noisy observation)

N = 150;
s1 = 1234; s2 = 8765;
w = zeros(N,1); v = zeros(N,1);
for t = 1:N
    for k = 1:5, s1 = mod(1103515245*s1 + 12345, 2147483648); end
    a = s1/2147483648;
    for k = 1:5, s1 = mod(1103515245*s1 + 12345, 2147483648); end
    b = s1/2147483648;
    if a < 1e-12, a = 1e-12; end
    w(t) = 0.2 * sqrt(-2*log(a)) * cos(2*pi*b);
    for k = 1:5, s2 = mod(1103515245*s2 + 12345, 2147483648); end
    c = s2/2147483648;
    for k = 1:5, s2 = mod(1103515245*s2 + 12345, 2147483648); end
    d = s2/2147483648;
    if c < 1e-12, c = 1e-12; end
    v(t) = 1.5 * sqrt(-2*log(c)) * cos(2*pi*d);
end
level = zeros(N,1);
level(1) = 5;
for t = 2:N, level(t) = level(t-1) + w(t); end
y = level + v;

% --- Build and estimate a local-level state-space model ----------------
A = ones(1,1); B = ones(1,1); C = ones(1,1); D = ones(1,1);
Mdl = ssm(A, B, C, D);
Est = estimate(Mdl, y);
fprintf('Estimated process std (B): %.3f\n', Est.B(1));
fprintf('Estimated obs std (D):     %.3f\n', Est.D(1));

% --- Smooth the latent level -------------------------------------------
xs = smooth(Est, y);

% Compare error of raw observations vs the smoothed estimate.
sse_obs = 0; sse_smooth = 0;
for t = 1:N
    sse_obs = sse_obs + (y(t) - level(t))^2;
    sse_smooth = sse_smooth + (xs(t) - level(t))^2;
end
fprintf('Obs SSE vs truth:    %.1f\n', sse_obs);
fprintf('Smooth SSE vs truth: %.1f\n', sse_smooth);

% --- Forecast the observation 10 steps ahead ---------------------------
yF = forecast(Est, 10, y);
fprintf('Forecast horizon:    %.0f\n', numel(yF));

fprintf('State-space Kalman smoothing complete.\n');
