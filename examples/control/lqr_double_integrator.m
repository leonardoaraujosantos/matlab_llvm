% lqr — linear-quadratic regulator on the canonical double integrator.
%
% Tier 3.1 (control_toolbox_roadmap.md §4.1) — NOT YET SHIPPED.
% Depends on Tier 1.5 (care — algebraic Riccati equation solver via
% the Hamiltonian / ordered Schur decomposition). lqr is a thin
% wrapper that calls care, computes K = R\B'*S, and returns the
% closed-loop poles.
%
% Plant: double integrator (mass under force).
%   xdot = [0 1; 0 0] x + [0; 1] u,  y = [1 0] x.
%   The two poles are at the origin (marginally unstable in the
%   integrator sense). LQR will move them into the LHP.

A = [0 1; 0 0];
B = [0; 1];
C = [1 0];
D = 0;

% --- 1. State-cost matrix Q penalises position more than velocity;
% control-cost R penalises actuation. Q = diag([10, 1]), R = 1.
Q = [10 0; 0 1];
R = 1;

% --- 2. Solve the LQR.
%   Returns:
%     K — feedback gain so u = -K*x stabilises the closed loop.
%     S — the unique stabilising solution of the algebraic Riccati eqn.
%     e — closed-loop poles (eigenvalues of A - B*K).
[K, S, e] = lqr(A, B, Q, R);
disp('LQR gain K:');
disp(K);
disp('closed-loop poles:');
disp(e);
% Symmetric-root-locus prediction:  the closed-loop poles are the
% stable roots of  R * det(s I - A) * det(-s I - A) + det(.. C'*Q*C ..)
% = 0.  For Q = diag(q1, q2), R = 1, double integrator, this is
% s^4 - q2*s^2 + q1 = 0.  With q1=10, q2=1: s^2 = (1 ± sqrt(1+40)) / 2
% = (1 ± sqrt(41))/2.  Stable s = -sqrt((sqrt(41)-1)/2 ± ...).
% Both poles should land in the open LHP.

% --- 3. Closed-loop simulation from a non-zero initial state.
%   x(0) = [1; 0], no input. Closed-loop:  xdot = (A - B*K) x.
%   Should drive x(1) → 0 with no overshoot in position (zeta > 1
%   case here is unlikely; expect a small overshoot).
Acl = A - B*K;
sys_cl = ss(Acl, zeros(2,1), C, 0);
t   = 0 : 0.01 : 10;
x0  = [1; 0];
y   = initial(sys_cl, x0, t);

disp('y(0):');
disp(y(1));        % position at t=0
disp('y(2 s):');
disp(y(201));      % should be small after 2 s
disp('y(end):');
disp(y(end));      % near 0 at t = 10 s

% --- 4. Discrete-time analog — dlqr.
Ts = 0.05;
sys_d = c2d(ss(A, B, C, D), Ts, 'zoh');
[Ad, Bd, Cd, Dd] = ssdata(sys_d);
[Kd, Sd, ed] = dlqr(Ad, Bd, Q, R);
disp('discrete-time LQR gain Kd:');
disp(Kd);
disp('discrete-time closed-loop poles (must lie inside unit circle):');
disp(abs(ed));
