% Tier 3 — H₂ system norm.
% norm_h2(A, B, C) = sqrt(trace(C · Wc · C')) where Wc = lyap(A, B B').
% The H₂ norm is the integral of |G(jw)|² over w (Parseval), which
% equals the integral of the impulse response squared. It's the system
% metric LQG cost is built on, and the natural "size" of a transfer
% function for noise-driven systems.

% --- 1. SISO 1st-order — closed form sanity.
% G(s) = 1/(s + 1); ||G||_2 = 1/sqrt(2) ≈ 0.7071.
A = [0-1];
B = [1];
C = [1];
fprintf('1st-order ||G||_2 = %.6f (closed form = %.6f)\n', ...
        norm_h2(A, B, C), 1/sqrt(2));

% --- 2. Sweep damping on a mass-spring-damper. ||G||_2 should drop
% as zeta increases (more damping = smaller integrated impulse energy).
wn = 3.0;
zlist = [0.05; 0.1; 0.2; 0.5; 1.0];
Bz = [0; 1];
Cz = [1, 0];
zeta = zlist(1, 1);
Az = [0, 1; 0-wn*wn, 0-2*zeta*wn];
fprintf('  zeta = %.2f, ||G||_2 = %.6f\n', zeta, norm_h2(Az, Bz, Cz));
zeta = zlist(2, 1);
Az = [0, 1; 0-wn*wn, 0-2*zeta*wn];
fprintf('  zeta = %.2f, ||G||_2 = %.6f\n', zeta, norm_h2(Az, Bz, Cz));
zeta = zlist(3, 1);
Az = [0, 1; 0-wn*wn, 0-2*zeta*wn];
fprintf('  zeta = %.2f, ||G||_2 = %.6f\n', zeta, norm_h2(Az, Bz, Cz));
zeta = zlist(4, 1);
Az = [0, 1; 0-wn*wn, 0-2*zeta*wn];
fprintf('  zeta = %.2f, ||G||_2 = %.6f\n', zeta, norm_h2(Az, Bz, Cz));
zeta = zlist(5, 1);
Az = [0, 1; 0-wn*wn, 0-2*zeta*wn];
fprintf('  zeta = %.2f, ||G||_2 = %.6f\n', zeta, norm_h2(Az, Bz, Cz));

% --- 3. The H₂ norm equals the H₂ cost of an LQR with Q = C'C, R = I,
% with closed-loop A_cl = A - B*K. Demonstrates that LQG cost = H₂
% norm of the closed-loop transfer function from process noise to
% performance output.
A3 = [0, 1; 0-9, 0-0.6];
B3 = [0; 1];
C3 = [1, 0];
fprintf('open-loop ||G||_2 = %.6f\n', norm_h2(A3, B3, C3));

Q  = C3' * C3;
R  = [1];
K  = lqr(A3, B3, Q, R);
Ac = A3 - B3 * K;
fprintf('after LQR, ||G_cl||_2 = %.6f (should be smaller)\n', ...
        norm_h2(Ac, B3, C3));

% --- 4. Unstable plant.
A4 = [1];
B4 = [1];
C4 = [1];
fprintf('unstable plant ||G||_2 = %.6f (must be +Inf)\n', ...
        norm_h2(A4, B4, C4));
