% recursive_arx_tracking.m — System Identification Toolbox Tier-5 headline.
%
% Online recursive estimation (User's Guide Ch.17/18: *Online ARX
% Parameter Estimation for Tracking Time-Varying System Dynamics*).  A
% plant pole shifts abruptly mid-experiment; a forgetting-factor RLS
% estimator follows it sample-by-sample with no batch re-fit.
%
%   recursiveARX([na nb nk]) -> step(obj, y, u) each sample
%
% The estimator object carries its mutable Parameters + Covariance and a
% buffered I/O history.

% ----- 1.  Time-varying plant -----------------------------------------
% y(t) = a(t) y(t-1) + u(t-1).  a = 0.5 for the first half, then 0.85.
N = 800;
u = zeros(N, 1); sd = 1234;
for k = 1:N
    sd = mod(sd*1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 2:N
    if k < 400, a = 0.50; else, a = 0.85; end
    y(k) = a*y(k-1) + 1.0*u(k-1);
end

% ----- 2.  Recursive ARX with a forgetting factor ---------------------
r = recursiveARX([1 1 1]);
r.ForgettingFactor = 0.96;        % < 1 → tracks change; 1 → infinite memory

% snapshot the estimate just before and well after the parameter jump
a_before = 0; a_after = 0;
th = [0; 0];
for k = 2:N
    th = step(r, y(k), u(k));
    if k == 390, a_before = -th(1); end   % A = [1 -a] → a = -th(1)
    if k == 800, a_after  = -th(1); end
end
fprintf('Tracked pole a(t):\n');
fprintf('  t=390 (before jump): a = %.3f   (true 0.50)\n', a_before);
fprintf('  t=800 (after  jump): a = %.3f   (true 0.85)\n', a_after);
fprintf('Final B coefficient:   b = %.3f   (true 1.00)\n', th(2));
