% lyap(A, Q) for the controllability gramian — Tier 1.4 (CST roadmap §2.4).
%
% The infinite-horizon controllability gramian Wc of a continuous-time
% LTI plant satisfies the Lyapunov equation
%      A Wc + Wc A' + B B' = 0
% with the unique solution Wc = lyap(A, B*B') for stable A. The
% observability gramian Wo solves the dual:
%      A' Wo + Wo A + C' C = 0  ->  Wo = lyap(A', C'*C).
% Together, the gramians underlie balanced realisation, model
% reduction (balred), and the H2 system norm.

% --- 1. Mass-spring-damper plant.
%   xdot = [0 1; -k/m -c/m] x + [0; 1/m] u,  y = [1 0] x.
m = 1.0; k = 9.0; c = 0.6;
A = [0 1; 0-k/m, 0-c/m];
B = [0; 1/m];
C = [1 0];

% --- 2. Controllability gramian via lyap.
Wc = lyap(A, B * B');
disp('Wc(1,1) and (2,2):');
disp(Wc(1, 1));     % position-energy
disp(Wc(2, 2));     % velocity-energy

% --- 3. Observability gramian.
Wo = lyap(A', C' * C);
disp('Wo(1,1) and (2,2):');
disp(Wo(1, 1));
disp(Wo(2, 2));

% --- 4. Lyapunov residual self-consistency. R = A Wc + Wc A' + B B'
% should be machine zero.
R = A * Wc + Wc * A' + B * B';
disp('residual entries (must be ~0 to round-off):');
disp(R(1, 1));
disp(R(2, 2));
disp(R(1, 2));

% --- 5. Discrete analogue. Discretise via expm (Tier 1.3) and use dlyap.
Ts = 0.05;
Ad = expm(A * Ts);
% Discrete controllability gramian: A_d X A_d' - X + Bd*Bd' = 0 (Bd
% derived from the augmented-matrix expm trick — using a simple
% approximation Bd = B * Ts here for brevity).
Bd = B * Ts;
Wcd = dlyap(Ad, Bd * Bd');
disp('discrete Wc(1,1) and (2,2):');
disp(Wcd(1, 1));
disp(Wcd(2, 2));

% --- 6. Stein residual. Should also be ~0.
Rd = Ad * Wcd * Ad' - Wcd + Bd * Bd';
disp('discrete residual entries:');
disp(Rd(1, 1));
disp(Rd(2, 2));
