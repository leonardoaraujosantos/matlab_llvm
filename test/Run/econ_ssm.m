% Econometrics Toolbox Tier-5 — ssm state-space (local level model).
%   level_t = level_{t-1} + w_t   (random walk state, A=1)
%   y_t     = level_t     + v_t   (noisy observation, C=1)

N = 200;
s1 = 4242; s2 = 9001;
w = zeros(N,1); v = zeros(N,1);
for t = 1:N
    for k = 1:5, s1 = mod(1103515245*s1 + 12345, 2147483648); end
    a = s1/2147483648;
    for k = 1:5, s1 = mod(1103515245*s1 + 12345, 2147483648); end
    b = s1/2147483648;
    if a < 1e-12, a = 1e-12; end
    w(t) = 0.3 * sqrt(-2*log(a)) * cos(2*pi*b);     % small process noise
    for k = 1:5, s2 = mod(1103515245*s2 + 12345, 2147483648); end
    c = s2/2147483648;
    for k = 1:5, s2 = mod(1103515245*s2 + 12345, 2147483648); end
    d = s2/2147483648;
    if c < 1e-12, c = 1e-12; end
    v(t) = 1.0 * sqrt(-2*log(c)) * cos(2*pi*d);     % larger obs noise
end
level = zeros(N,1);
level(1) = 10;
for t = 2:N, level(t) = level(t-1) + w(t); end
y = level + v;

% --- Build a local-level ssm and estimate the noise std devs ------------
% (1x1 system matrices built with ones() so they carry the matrix type).
A = ones(1,1); B = ones(1,1); C = ones(1,1); D = ones(1,1);
Mdl = ssm(A, B, C, D);
Est = estimate(Mdl, y);
fprintf('kind = %.0f\n', Est.ModelKind);        % 6

% --- Kalman filter + RTS smoother extract the latent level --------------
xf = filter(Est, y);
xs = smooth(Est, y);
fprintf('nfilt = %.0f\n', numel(xf));            % 200
fprintf('nsmooth = %.0f\n', numel(xs));          % 200

% Smoothing should track the true level better than the raw observations:
% report both sums of squared errors (smoothing SSE should be smaller).
ey = 0; es = 0;
for t = 1:N
    ey = ey + (y(t) - level(t))^2;
    es = es + (xs(t) - level(t))^2;
end
fprintf('obs_sse = %.1f\n', ey);     % raw observation error
fprintf('smooth_sse = %.1f\n', es);  % smoothing error (should be smaller)

% --- Forecast 5 steps ahead --------------------------------------------
yF = forecast(Est, 5, y);
fprintf('nfc = %.0f\n', numel(yF));              % 5
