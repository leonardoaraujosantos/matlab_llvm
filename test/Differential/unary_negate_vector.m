% Unary float negation inside a vector literal and an expression.
u = 3.5;
v = [u; -u; -2*u; -(u+1)];
disp(v);
fprintf('neg=%.4f\n', -u);
