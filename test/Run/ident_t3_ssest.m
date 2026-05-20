% System Identification Tier-3 — subspace state-space estimation.
% True 2nd-order plant y(t)=1.5y(t-1)-0.7y(t-2)+1.0u(t-1)+0.5u(t-2).
% ssest recovers an order-2 model; trace/det of A are similarity
% invariants = sum/product of the poles (0.75±0.36i → trace 1.5, det 0.70).
N = 600;
u = zeros(N, 1); sd = 271828;
for k = 1:N
    sd = mod(sd * 1103515245 + 12345, 2147483648);
    u(k) = sign(sd/2147483648 - 0.5);
end
y = zeros(N, 1);
for k = 3:N
    y(k) = 1.5*y(k-1) - 0.7*y(k-2) + 1.0*u(k-1) + 0.5*u(k-2);
end
z = iddata(y, u, 0.1);
sys = ssest(z, 2);
fprintf('nx = %.0f\n', size(sys.A, 1));      % 2
fprintf('Ts = %.2f\n', sys.Ts);             % 0.10
fprintf('fit = %.1f\n', compare(z, sys));    % ~96.8
fprintf('trace = %.3f\n', trace(sys.A));     % 1.500 (sum of poles)
fprintf('det = %.3f\n', det(sys.A));         % 0.700 (product of poles)
