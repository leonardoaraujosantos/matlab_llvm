% Tier 2 — full SISO interconnection workflow without model objects.
% Demonstrates the four matrix-arg primitives shipped this session:
%   series_ss   — sys = sys2 * sys1
%   parallel_ss — sys = sys1 + sys2
%   feedback_ss — sys = sys1 / (1 + sys2 * sys1)   (negative feedback)
%   append_ss   — sys = blkdiag(sys1, sys2)         (MIMO assembly)
% All require strictly-proper plants (D1 = D2 = 0).

% --- Plant: 2nd-order mass-spring-damper.
% xdot = [0 1; -wn^2 -2*zeta*wn] x + [0; 1] u, y = [1 0] x.
wn   = 3.0;
zeta = 0.1;
A1 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B1 = [0; 1];
C1 = [1, 0];

% --- Compensator: PI in 1-state realization. xc' = e (error), y = xc.
A2 = [0];
B2 = [1];
C2 = [1];

% --- 1. SERIES — controller before plant: G(s) · K(s).
[As, Bs, Cs] = series_ss(A2, B2, C2, A1, B1, C1);
disp('series Acl (3 x 3):');
disp(As);
disp('series eig:');
disp(real(eig(As)));

% --- 2. PARALLEL — feedforward + plant in parallel.
[Ap, Bp, Cp] = parallel_ss(A1, B1, C1, A2, B2, C2);
disp('parallel Acl (3 x 3):');
disp(Ap);
disp('parallel Bcl (3 x 1, stacked):');
disp(Bp);
disp('parallel Ccl (1 x 3, summed):');
disp(Cp);

% --- 3. FEEDBACK — close the loop. T = sys_open / (1 + sys_open).
% Build the open-loop sys_open = sys2 * sys1 (compensator feeding plant)
% then close negative feedback against unit gain (sys_fb = constant 1
% via a stable 1-state realization with ε leak so it's strictly proper:
% A = -1e6, B = 1e6, C = 1 → high-bandwidth tracker that settles fast).
% In practice users close the loop with sys_fb = identity in the model-
% object form. Here we use the 2-state open-loop with a static-gain
% feedback approximation.
[Aol, Bol, Col] = series_ss(A2, B2, C2, A1, B1, C1);
% Negative feedback against unity (high-bw 1-state proxy).
A_fb = [0-1e3]; B_fb = [1e3]; C_fb = [1];
[Acl, Bcl, Ccl] = feedback_ss(Aol, Bol, Col, A_fb, B_fb, C_fb);
disp('feedback Acl (4 x 4 = 3+1):');
disp(Acl);
disp('feedback isstable (closed-loop must be Hurwitz):');
disp(isstable(Acl));

% --- 4. APPEND — block-diagonal MIMO. Two SISO plants stacked into
% a 2-input 2-output system with no cross-coupling. Useful when
% assembling decoupled MIMO models for multivariable design.
[Aa, Ba, Ca] = append_ss(A1, B1, C1, A1, B1, C1);
disp('append Acl (4 x 4, blkdiag):');
disp(Aa);
disp('append Bcl (4 x 2, blkdiag — disjoint input channels):');
disp(Ba);
disp('append Ccl (2 x 4, blkdiag — disjoint output channels):');
disp(Ca);
