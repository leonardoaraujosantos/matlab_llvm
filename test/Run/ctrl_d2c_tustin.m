% Tier 2.2 — inverse Tustin: [A, B] = d2c_tustin(Ad, Bd, Ts).
% Pair to c2d_tustin. c2d → d2c is an exact round-trip to machine
% precision since both are closed-form rational maps.

% --- 1. Round-trip on a stable 2-state plant.
A = [0-1, 0.5; 0, 0-2];
B = [1; 0.5];
Ts = 0.1;

[Ad, Bd] = c2d_tustin(A, B, Ts);
[A2, B2] = d2c_tustin(Ad, Bd, Ts);

disp('original A:');
disp(A);
disp('round-trip A2 (must match):');
disp(A2);
disp('original B:');
disp(B);
disp('round-trip B2 (must match):');
disp(B2);

% --- 2. 1-return form returns A.
A_only = d2c_tustin(Ad, Bd, Ts);
disp('1-return form A:');
disp(A_only);
