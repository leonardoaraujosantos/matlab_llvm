% Tier 4 — balanced realization workflow.
% T = balreal_T(A, B, C) returns the similarity transformation that
% puts an LTI system into internally-balanced form: the controllability
% and observability gramians become equal and diagonal, with diagonal
% entries equal to the Hankel singular values (descending).
%
% Balanced realization is the structural foundation for `balred` (model
% reduction by truncating the smallest HSVs).

% --- 1. Open-loop mass-spring-damper plant.
% xdot = [0 1; -wn^2 -2*zeta*wn] x + [0; 1] u, y = [1 0] x.
wn = 3.0; zeta = 0.05;
A = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B = [0; 1];
C = [1, 0];

H = hsvd(A, B, C);
disp('Hankel singular values (descending):');
disp(H);

% --- 2. Compute the balancing transform.
T  = balreal_T(A, B, C);
Ti = inv(T);

% --- 3. Construct the balanced realization.
Ab = Ti * A * T;
Bb = Ti * B;
Cb = C  * T;

disp('balanced A:');
disp(Ab);

% --- 4. Verify the balanced gramians equal diag(HSV).
Wcb = gram_c(Ab, Bb);
Wob = gram_o(Ab, Cb);
disp('balanced Wc (must = diag(HSV)):');
disp(Wcb);
disp('balanced Wo (must = diag(HSV)):');
disp(Wob);
disp('Wcb - Wob (must be ~0 — the balanced invariant):');
disp(Wcb - Wob);

% --- 5. The transfer function is invariant under similarity, so the
% open-loop and balanced realizations have the same impulse / step /
% bode response. The point of balancing is *internal* — it makes the
% state ordering reflect importance for the I/O map, so truncating
% small-HSV states gives a low-order model with controlled error.
disp('hsvd(Ab, Bb, Cb) — must equal hsvd(A, B, C):');
disp(hsvd(Ab, Bb, Cb));

% ----- plot the Hankel singular values -------------------------------
figure; bar(H); grid on;
xlabel('state'); ylabel('Hankel SV'); title('balanced realization HSVs');
saveas(gcf, '/tmp/ctrl_balreal_hsv.png');
