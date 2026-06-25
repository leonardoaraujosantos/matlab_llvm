% Nested matrix subexpression as operand of scalar multiply (precedence).
A = [1; 2; 3];
B = [10; 20; 30];
C = [100; 200; 300];
R = A + 0.5 * (B + C);
disp(R);
