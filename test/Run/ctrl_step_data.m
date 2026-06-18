% #322 regression: the data-returning [y, t] = step(sys) form (no time
% vector supplied) must return the response y and the auto-generated
% time grid t, not just plot. Before the fix this raised
% "unsupported call shape for built-in function 'step': 1 argument,
% 2 return values".
%
% First-order plant H(s) = 1/(s+1): unit step response y(t) = 1 - e^(-t),
% sampled on the default grid (dt = 0.01, N = 500, so t = 0 .. 4.99).

sys = tf([1], [1 1]);
[y, t] = step(sys);
fprintf('ny = %d\n', numel(y));
fprintf('nt = %d\n', numel(t));
fprintf('t0 = %.2f\n', t(1));
fprintf('tend = %.2f\n', t(end));
fprintf('y0 = %.4f\n', y(1));
fprintf('yend = %.4f\n', y(end));

% Same plant as a state-space realisation — the ss path must agree.
sys2 = ss(-1, 1, 1, 0);
[y2, t2] = step(sys2);
fprintf('ss_n = %d\n', numel(y2));
fprintf('ss_yend = %.4f\n', y2(end));
fprintf('ss_tend = %.2f\n', t2(end));
