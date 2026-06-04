% norm(x, p): p-norm with explicit order (#feature). Vector p=1/2/Inf and the
% cheap matrix induced norms p=1 (max col abs-sum) / p=Inf (max row abs-sum).
x = [3 4];
fprintf('vec: %.1f %.1f %.1f\n', norm(x,1), norm(x,2), norm(x,Inf));
y = [1 -2 3 -4];
fprintf('vec2: %.1f %.1f\n', norm(y,1), norm(y,Inf));
A = [1 -2; -3 4];
fprintf('mat: %.1f %.1f\n', norm(A,1), norm(A,Inf));
fprintf('def: %.1f\n', norm(x));
