% Solve Ax = b using MATLAB's left-divide operator.
%
%   2x + y +  z = 5
%    x + 3y + 2z = 10
%    x +  y +  z = 6
A = [2 1 1;
     1 3 2;
     1 1 1];
b = [5; 10; 6];

x = A \ b;

disp('A =');
disp(A);
disp('b =');
disp(b);
disp('x = A \ b =');
disp(x);

% Verify: A * x should reproduce b.
disp('A * x =');
disp(A * x);
