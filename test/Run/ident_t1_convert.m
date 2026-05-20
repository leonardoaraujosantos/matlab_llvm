% System Identification Tier-1 — idpoly → ss / tf conversion.
% ss(model) builds the controllable-canonical realisation carrying the
% discrete Ts; tf(model) extracts B/A.  CST `pole` then reuses the ss.
N = 200;
u = zeros(N, 1);
for k = 1:N
    u(k) = sin(0.3 * k);
end
y = zeros(N, 1);
for k = 2:N
    y(k) = 0.5 * y(k-1) + 1.0 * u(k-1);
end
z = iddata(y, u, 1);
m = arx(z, [1 1 1]);

sys = ss(m);
fprintf('ss_Ts  = %.2f\n', sys.Ts);       % 1.00 (discrete)
fprintf('ss_A   = %.4f\n', sys.A(1,1));    % 0.5000
fprintf('ss_C   = %.4f\n', sys.C(1,1));    % 1.0000
fprintf('ss_D   = %.4f\n', sys.D(1,1));    % 0.0000
p = pole(sys);
fprintf('pole   = %.4f\n', p(1));          % 0.5000

g = tf(m);
fprintf('tf_num = %.4f\n', g.Numerator(2));    % 1.0000
fprintf('tf_den = %.4f\n', g.Denominator(2));  % -0.5000
