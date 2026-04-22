A = zeros(3, 4);
B = ones(4, 5);
C = A * B;          % 3x5 double
D = A';             % 4x3 double
E = [1 2 3];
F = [1; 2; 3];
G = E + 1;          % scalar broadcast, vec
H = A + 2.0;        % scalar broadcast, shape = A
I = 1:5;            % vec
