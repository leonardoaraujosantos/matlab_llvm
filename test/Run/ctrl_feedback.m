% Tier 2 — `feedback_ss(A1, B1, C1, A2, B2, C2)` matrix-arg closed-loop
% assembly for negative feedback (strictly proper plants D1 = D2 = 0).
%
%   Acl = [A1, -B1*C2; B2*C1, A2]
%   Bcl = [B1; 0]
%   Ccl = [C1, 0]

% --- 1. Plant (1st-order) + integrator compensator. Closed-loop must
% be Hurwitz.
A1 = [0-1, 1; 0, 0-2];   % 2-state plant
B1 = [0; 1];
C1 = [1, 0];
A2 = [0];                % integrator compensator
B2 = [1];
C2 = [1];

[Acl, Bcl, Ccl] = feedback_ss(A1, B1, C1, A2, B2, C2);
disp('Acl (3 x 3 = n1+n2):');
disp(Acl);
disp('Bcl (3 x 1):');
disp(Bcl);
disp('Ccl (1 x 3):');
disp(Ccl);
disp('isstable(Acl):');
disp(isstable(Acl));

% --- 2. 1-return form: defaults to Acl.
A_only = feedback_ss(A1, B1, C1, A2, B2, C2);
disp('1-return Acl (must match):');
disp(A_only);

% --- 3. Both plants 1×1 — minimal example. sys1 = sys2 = 1/(s+1).
% Closed-loop: 1/(s+1) / (1 + 1/(s+1)^2). State-space layout:
% Acl = [-1, -1; 1, -1]; eigvals = -1 ± j.
A3 = [0-1]; B3 = [1]; C3 = [1];
[A3cl, B3cl, C3cl] = feedback_ss(A3, B3, C3, A3, B3, C3);
disp('1×1 each, closed-loop Acl (eig = -1 ± j):');
disp(A3cl);
disp('eig real parts (must both be -1):');
disp(real(eig(A3cl)));
disp('eig imag parts (must be ±1):');
disp(imag(eig(A3cl)));
