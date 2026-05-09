% Tier 2.2 — Tustin (bilinear) c2d.
% [Ad, Bd] = c2d_tustin(A, B, Ts) — substitutes s = (2/Ts)·(z−1)/(z+1).
% No expm needed; closed-form rational map of the s-plane to the
% z-plane. Stable continuous plants stay stable in discrete.

% --- 1. SISO 1×1 closed form. a = -1, b = 1, Ts = 0.1, α = 0.05.
% Ad = (1 + α a)/(1 - α a) = 0.95/1.05 = 0.9048
% Bd = Ts b /(1 - α a) = 0.1/1.05 = 0.0952
A = [0-1];
B = [1];
[Ad, Bd] = c2d_tustin(A, B, 0.1);
fprintf('1×1 Ad (closed form 0.904762): %.6f\n', Ad(1, 1));
fprintf('1×1 Bd (closed form 0.095238): %.6f\n', Bd(1, 1));

% --- 2. Mass-spring-damper, Hurwitz → Schur preserved.
wn = 3.0; zeta = 0.5;
A2 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B2 = [0; 1];
[A2d, B2d] = c2d_tustin(A2, B2, 0.05);
fprintf('isstable (continuous): %.0f\n', isstable(A2));
fprintf('isstable_d after Tustin: %.0f\n', isstable_d(A2d));

disp('Tustin Ad (must be Schur):');
disp(A2d);
disp('Tustin Bd:');
disp(B2d);

% --- 3. Compare ZOH vs Tustin on the same plant. They differ; both
% should be Schur-stable for a stable continuous plant.
[Az, Bz] = c2d(A2, B2, 0.05);
disp('ZOH Ad:');
disp(Az);
disp('Tustin Ad:');
disp(A2d);
disp('ZOH eig:');
disp(real(eig(Az)));
disp('Tustin eig:');
disp(real(eig(A2d)));
