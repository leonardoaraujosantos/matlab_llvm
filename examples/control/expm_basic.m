% Matrix exponential expm(A) — the workhorse primitive of the Control
% System Toolbox numeric stack.
%
% Tier 1.3 (control_toolbox_roadmap.md §2.3) — shipped.
%
% expm shows up in:
%   - c2d ZOH discretisation: Ad = expm(A*Ts), and the augmented-matrix
%     trick gives Bd in the same call.
%   - lsim continuous-time exact step: x(t+dt) = expm(A*dt)*x(t) + ...
%   - initial-condition response: x(t) = expm(A*t) * x0.
%   - Lyapunov / Riccati closed-form transcriptions.

% --- 1. The defining identity: expm(0) = I.
A0 = zeros(3, 3);
E0 = expm(A0);
disp('expm(zeros(3,3)) — diagonal entries:');
disp(E0(1, 1));
disp(E0(2, 2));
disp(E0(3, 3));

% --- 2. Rotation matrix.
%   A = [0  1; -1  0]  is the generator of 2-D rotation;
%   expm(A * theta) = [cos(theta) sin(theta); -sin(theta) cos(theta)].
%
% Verify at theta = pi/2:  expm(A * pi/2) = [0 1; -1 0] (the matrix A itself).
A = [0 1; -1 0];
R = expm(A * (pi/2));
disp('rotation by pi/2 — entries:');
disp(R(1, 1));   % cos(pi/2) = 0
disp(R(1, 2));   % sin(pi/2) = 1
disp(R(2, 1));   % -sin(pi/2) = -1
disp(R(2, 2));   % cos(pi/2) = 0

% --- 3. Free response of a stable LTI system.
%   xdot = A * x, x(0) = x0  ⇒  x(t) = expm(A * t) * x0.
%   Pick A with eigenvalues at -1 and -2; the response decays.
%
% A = [-1 0; 0 -2], x0 = [1; 1] ⇒  x(1) = [exp(-1); exp(-2)] ≈ [0.368; 0.135].
A2 = [-1 0; 0 -2];
x0 = [1; 1];
x1 = expm(A2 * 1.0) * x0;
disp('free response x(t=1) for diag([-1 -2]):');
disp(x1(1));    % exp(-1) ≈ 0.36787944117
disp(x1(2));    % exp(-2) ≈ 0.13533528324

% --- 4. The c2d ZOH augmented-matrix trick.
%   For xdot = A*x + B*u with zero-order hold on u,
%       expm([A B; 0 0] * Ts) = [Ad Bd; 0 I].
%   This is how c2d will be implemented in Tier 2.2 — one expm call
%   gives both Ad and Bd at once.
% Build the augmented matrix
%   M = [ A3   B3 ;
%         0  0  0 ]
% explicitly as a 3x3 literal because the matrix-of-matrices
% concatenation form `[A3 B3; 0 0 0]` is still gated on a concat-row
% lowering generalisation (mixed scalar/matrix rows). Equivalent.
Ts = 0.1;
M  = [-1 0 1; 0 -2 0.5; 0 0 0];
EM = expm(M * Ts);
% Top-left 2x2 block is Ad; top-right 2x1 column is Bd.
disp('discretised Ad(1,1):');
disp(EM(1, 1));        % exp(-1*0.1) = 0.9048374...
disp('discretised Ad(2,2):');
disp(EM(2, 2));        % exp(-2*0.1) = 0.8187307...
disp('discretised Bd(1):');
disp(EM(1, 3));        % integral of exp(-tau)*1 from 0..0.1 = 0.0951625...
disp('discretised Bd(2):');
disp(EM(2, 3));        % 0.5 * (1 - exp(-0.2)) / 2 = 0.0453173...
