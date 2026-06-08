% Tier 3 — pole placement workflow.
% The user-facing alternative to LQR: instead of optimising a quadratic
% cost, the engineer specifies *where the closed-loop poles should go*
% and `place` (or `acker`) computes the gain that puts them there.
%
% Flow:
%   1. Confirm controllability via rank(ctrb(A, B)) = n.
%   2. Pick desired closed-loop poles P.
%   3. K = place(A, B, P).
%   4. Closed-loop A − B K has eig = P (sorted ascending).
%
% This example uses the inverted pendulum near the upright equilibrium —
% one of the classic teaching plants where root-locus design becomes
% intuitive.

% --- 1. Inverted pendulum, linearised at theta = 0 (upright).
% State x = [theta; theta_dot]. Open-loop is unstable: real positive
% eigenvalue at +sqrt(g/L) ≈ +3.13 rad/s.
g = 9.81; L = 1.0;
A = [0 1; g/L, 0];
B = [0; 1];
disp('open-loop eig (one positive — pendulum falls):');
disp(real(eig(A)));

% Controllability test.
Co = ctrb(A, B);
disp('ctrb(A, B):');
disp(Co);
disp('det(Co) (must be nonzero for controllability):');
disp(det(Co));

% --- 2. Desired closed-loop spec: lightly-damped well-placed poles
% at  -2 +- 2j  (natural freq ≈ 2.83, damping = 0.707).
P = [0 - 2.0; 0 - 2.0];     % Two real poles at -2 (Ackermann handles repeated)
K = place(A, B, P);
disp('place gain K (state feedback):');
disp(K);

% --- 3. Verify closed-loop assignment.
Acl = A - B * K;
disp('eig(A - B K) (must equal P):');
disp(real(eig(Acl)));

% --- 4. The energy-based companion: gram_c on the *closed-loop* plant
% (must be PD since Acl is Hurwitz). Compares the structural rank
% (ctrb full) and the energy gramian; both confirm controllability.
Wc = gram_c(Acl, B);
disp('controllability gramian of (Acl, B):');
disp(Wc);
disp('Wc(1,1) > 0 (PD diagonal):');
disp(Wc(1,1) > 0);

% --- 5. Observability with C = [1 0] (measure theta only). The pair
% (A, C) must be observable for a state estimator to work.
C = [1, 0];
Ob = obsv(A, C);
disp('obsv(A, C):');
disp(Ob);
disp('det(Ob) (nonzero → observable):');
disp(det(Ob));

% ----- plot open-loop vs placed closed-loop poles --------------------
eo = eig(A); ecl = eig(Acl);
re = [real(eo); real(ecl)]; im = [imag(eo); imag(ecl)];
figure; scatter(real(eo), imag(eo)); hold on; scatter(real(ecl), imag(ecl));
plot([0 0], [min(im)-1, max(im)+1], 'k-');     % stability boundary (Im axis)
grid on; axis([min(re)-1, max(max(re)+1, 0.5), min(im)-1, max(im)+1]);
xlabel('Re'); ylabel('Im');
title('poles: open-loop (one in RHP) placed into the LHP');
legend('open-loop', 'closed-loop');
saveas(gcf, '/tmp/ctrl_place.png');
