% A const-valued scalar (pi, 2*pi) times a function-of-a-vector inside a sum
% must keep the vector shape (previously collapsed to a scalar -> lowering bail).
m = (1:6)';
y = 1.5 + 2.0*sin(2*pi*m/12) + 0.8*cos(2*pi*m/12);
fprintf('%.4f\n', y(1));
fprintf('%.4f\n', y(4));
z = pi*m;
fprintf('%.4f\n', z(2));
