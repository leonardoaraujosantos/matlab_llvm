% State feedback u = -K*x driving a discrete linear system in a loop.
K = [2.0, 0.5];
Ad = [1.0, 0.1; 0.0, 1.0];
Bd = [0.0; 0.1];
x = [1.0; 0.0];
for t = 1:25
    u = -K * x;
    x = Ad * x + Bd * u;
end
fprintf('x1=%.6f x2=%.6f\n', x(1), x(2));
